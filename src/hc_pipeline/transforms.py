"""Pure DataFrame transformations: raw CSV -> bronze -> silver (+ quarantine).

Nothing in this module reads configuration or writes tables, so every step can be
unit-tested with a local SparkSession (see tests/).
"""
from __future__ import annotations

import csv
import io

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .spec import DAYS_PLACEHOLDER, META_COLUMNS, TableSpec, resolve_type

CORRUPT = "_corrupt_record"


# --------------------------------------------------------------------------- bronze
def read_header(spark: SparkSession, path: str) -> list[str]:
    """Read the CSV header line and return the column names (BOM-safe)."""
    first = spark.read.text(path).limit(1).collect()
    if not first:
        raise ValueError(f"Empty file: {path}")
    line = first[0][0].lstrip("﻿")
    return next(csv.reader(io.StringIO(line)))


def read_raw_csv(spark: SparkSession, path: str) -> DataFrame:
    """Read a raw CSV with every business column as STRING.

    Strings keep the raw values intact (no silent coercion); typing happens in silver
    where failures can be counted. Lines with the wrong number of fields are kept in
    `_corrupt_record` (PERMISSIVE mode) instead of being dropped.
    """
    columns = read_header(spark, path)
    schema = T.StructType([T.StructField(c, T.StringType()) for c in columns]
                          + [T.StructField(CORRUPT, T.StringType())])
    return (spark.read.schema(schema)
            .option("header", True)
            .option("mode", "PERMISSIVE")
            .option("columnNameOfCorruptRecord", CORRUPT)
            .option("multiLine", False)
            .csv(path))


def to_bronze(raw: DataFrame, source_version: str, source_sha256: str | None,
              run_id: str) -> DataFrame:
    """Add ingestion metadata to a raw string DataFrame."""
    return (raw
            .withColumn("_source_file", F.col("_metadata.file_path"))
            .withColumn("_source_version", F.lit(source_version))
            .withColumn("_source_sha256", F.lit(source_sha256).cast("string"))
            .withColumn("_ingest_run_id", F.lit(run_id))
            .withColumn("_bronze_row_id", F.monotonically_increasing_id()))


# --------------------------------------------------------------------------- silver
def business_columns(df: DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]


def _typed_and_violation(col: str):
    """Return (typed_expr, violation_expr) for one raw string column."""
    raw = F.col(f"`{col}`")
    target = resolve_type(col)
    if target == "string":
        return raw, F.lit(False)
    as_double = F.expr(f"try_cast(`{col}` AS DOUBLE)")
    bad = raw.isNotNull() & (as_double.isNull() | F.isnan(as_double)
                             | (F.abs(as_double) == float("inf")))
    if target in ("int", "bigint"):
        bad = bad | (as_double.isNotNull() & (as_double != F.floor(as_double)))
        typed = F.when(~bad, as_double.cast(target))
    else:
        typed = F.when(~bad, as_double)
    return typed, bad


def validate(bronze: DataFrame, spec: TableSpec) -> tuple[DataFrame, list[str]]:
    """Trim, type and flag every row.

    Returns the flagged DataFrame (typed business columns + `_raw` struct of the
    original strings + `_reasons` array) and the list of required columns missing
    from the file. Missing columns are added as NULL so the run can finish and the
    quality gate can report a schema failure instead of crashing.
    """
    cols = business_columns(bronze)
    missing = [c for c in spec.required_columns if c not in cols]
    df = bronze
    for c in missing:
        df = df.withColumn(c, F.lit(None).cast("string"))
    cols = cols + missing

    # 1) trim + empty string -> NULL (raw values are kept for the quarantine record)
    trimmed = df.select(
        *[F.when(F.trim(F.col(f"`{c}`")) == "", None).otherwise(F.trim(F.col(f"`{c}`"))).alias(c)
          for c in cols],
        *[c for c in META_COLUMNS if c in df.columns])

    # 2) cast + type-violation flags
    typed_cols, reason_exprs = [], []
    for c in cols:
        typed, bad = _typed_and_violation(c)
        typed_cols.append(typed.alias(c))
        reason_exprs.append(F.when(bad, F.lit(f"type:{c}")))
    if CORRUPT in trimmed.columns:
        reason_exprs.append(F.when(F.col(CORRUPT).isNotNull(), F.lit("malformed_line")))

    meta_present = [c for c in META_COLUMNS if c in trimmed.columns and c != CORRUPT]
    typed_df = trimmed.select(
        *typed_cols,
        F.struct(*[F.col(f"`{c}`") for c in cols]).alias("_raw"),
        F.array(*reason_exprs).alias("_type_reasons"),
        *meta_present)

    # 3) business rules evaluated on typed values
    rule_exprs = [F.when(F.coalesce(F.expr(r.violation_sql), F.lit(False)), F.lit(r.name))
                  for r in spec.rules]
    flagged = typed_df.withColumn(
        "_reasons",
        F.filter(F.concat(F.col("_type_reasons"), F.array(*rule_exprs) if rule_exprs
                          else F.array().cast("array<string>")),
                 lambda x: x.isNotNull())
    ).drop("_type_reasons")
    return flagged, missing


def deduplicate(flagged: DataFrame, spec: TableSpec, cols: list[str]) -> DataFrame:
    """Flag duplicates among rows that passed validation.

    * exact duplicate rows (all business columns equal): keep the first, flag the rest
      as `duplicate_row`;
    * remaining rows that share a primary key but differ in content: flag all of them
      as `duplicate_key_conflict` (we cannot tell which version is correct).
    """
    valid = F.size("_reasons") == 0
    order = F.col("_bronze_row_id") if "_bronze_row_id" in flagged.columns else F.lit(0)
    w_exact = Window.partitionBy(*[F.col(f"`{c}`") for c in cols]).orderBy(order)
    df = flagged.withColumn(
        "_dup_rank", F.when(valid, F.row_number().over(w_exact)))
    df = df.withColumn("_reasons", F.when(F.col("_dup_rank") > 1,
                                          F.array_append("_reasons", F.lit("duplicate_row")))
                       .otherwise(F.col("_reasons")))
    if spec.primary_key:
        still_valid = F.size("_reasons") == 0
        w_key = Window.partitionBy(*[F.col(k) for k in spec.primary_key])
        df = df.withColumn("_key_n", F.sum(F.when(still_valid, 1).otherwise(0)).over(w_key))
        df = df.withColumn("_reasons", F.when(still_valid & (F.col("_key_n") > 1),
                                              F.array_append("_reasons", F.lit("duplicate_key_conflict")))
                           .otherwise(F.col("_reasons"))).drop("_key_n")
    return df.drop("_dup_rank")


def clean(accepted: DataFrame, spec: TableSpec) -> DataFrame:
    """Apply value-level cleaning to accepted rows. The columns changed in each row are
    listed in `_cleaned_cols`, so every modification stays traceable."""
    actions = [F.when(F.col(c) == DAYS_PLACEHOLDER, F.lit(c))
               for c in spec.placeholder_null_columns if c in accepted.columns]
    actions += [F.when(F.col(c).isin(*v), F.lit(c))
                for c, v in spec.category_null_values.items() if c in accepted.columns]
    cleaned = (F.filter(F.array(*actions), lambda x: x.isNotNull()) if actions
               else F.array().cast("array<string>"))
    return _apply_clean_from(accepted.withColumn("_cleaned_cols", cleaned), spec)


def _apply_clean_from(flags: DataFrame, spec: TableSpec) -> DataFrame:
    df = flags
    if "DAYS_EMPLOYED" in spec.placeholder_null_columns and "DAYS_EMPLOYED" in df.columns:
        df = df.withColumn("DAYS_EMPLOYED_ANOM",
                           F.coalesce(F.col("DAYS_EMPLOYED") == DAYS_PLACEHOLDER, F.lit(False)).cast("int"))
    for c in spec.placeholder_null_columns:
        if c in df.columns:
            df = df.withColumn(c, F.when(F.col(c) == DAYS_PLACEHOLDER, None).otherwise(F.col(c)))
    for c, values in spec.category_null_values.items():
        if c in df.columns:
            df = df.withColumn(c, F.when(F.col(c).isin(*values), None).otherwise(F.col(c)))
    return df


def bronze_to_silver(bronze: DataFrame, spec: TableSpec, run_id: str) -> dict:
    """Full silver step. Returns {'accepted', 'rejected', 'missing_columns', 'columns'}."""
    flagged, missing = validate(bronze, spec)
    cols = [c for c in flagged.columns if not c.startswith("_")]
    flagged = deduplicate(flagged, spec, cols)

    ok = F.size("_reasons") == 0
    keep_meta = [c for c in ("_source_file", "_source_version") if c in flagged.columns]
    accepted = clean(flagged.filter(ok).select(*cols, *keep_meta), spec)
    accepted = accepted.withColumn("_silver_run_id", F.lit(run_id))

    rejected = flagged.filter(~ok).select(
        F.lit(spec.name).alias("table_name"),
        F.col("_source_version").alias("source_version") if "_source_version" in flagged.columns
        else F.lit(None).cast("string").alias("source_version"),
        F.col("_source_file").alias("source_file") if "_source_file" in flagged.columns
        else F.lit(None).cast("string").alias("source_file"),
        F.col("_bronze_row_id").alias("bronze_row_id") if "_bronze_row_id" in flagged.columns
        else F.lit(None).cast("bigint").alias("bronze_row_id"),
        F.col("_reasons").alias("reasons"),
        F.to_json("_raw", {"ignoreNullFields": "false"}).alias("raw_record"),
        F.lit(run_id).alias("run_id"),
        F.current_timestamp().alias("rejected_at"),
    )
    return {"accepted": accepted, "rejected": rejected, "missing_columns": missing, "columns": cols}
