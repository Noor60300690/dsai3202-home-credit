"""Row accounting and the quality gate.

`table_metrics` measures one staged silver table; `evaluate_table` and
`evaluate_cross_table` turn metrics into check results. A failed BLOCKING check
stops publication (the job task raises `QualityGateError`); WARN checks are recorded
but do not stop the run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .spec import TableSpec

BLOCK, WARN = "BLOCK", "WARN"

# Thresholds (kept here, not hidden in code paths, so the README can cite them).
TARGET_RATE_RANGE = (0.05, 0.12)       # real data: ~8.07 % positives
EXPECTED_ROWS_TOLERANCE = 0.01          # +/- 1 % of the documented row count
HIGH_NULL_RATE = 0.50                   # warn when a required column is mostly empty
MAX_ORPHAN_RATE = 0.50                  # warn when many child rows have no parent


class QualityGateError(RuntimeError):
    """Raised when at least one BLOCKING check fails."""


@dataclass
class CheckResult:
    table_name: str
    check: str
    severity: str
    passed: bool
    observed: str
    threshold: str

    def as_row(self) -> dict:
        return asdict(self)


def table_metrics(spec: TableSpec, bronze_rows: int, accepted: DataFrame, rejected: DataFrame,
                  missing_columns: list[str]) -> dict:
    """Measure a staged table. `accepted`/`rejected` should already be persisted
    (Delta tables on Databricks) so each count is a cheap scan."""
    m: dict = {"table_name": spec.name, "rows_in": int(bronze_rows),
               "missing_columns": list(missing_columns)}
    aggs = [F.count(F.lit(1)).alias("n")]
    for c in spec.required_columns:
        if c in accepted.columns:
            aggs.append(F.sum(F.col(f"`{c}`").isNull().cast("int")).alias(f"null__{c}"))
    if "TARGET" in accepted.columns:
        aggs.append(F.avg("TARGET").alias("target_rate"))
    row = accepted.agg(*aggs).first().asDict()
    m["rows_accepted"] = int(row.pop("n"))
    m["target_rate"] = row.pop("target_rate", None)
    n = max(m["rows_accepted"], 1)
    m["null_rate"] = {k[6:]: (v or 0) / n for k, v in row.items()}

    m["rows_rejected"] = rejected.count()
    m["reject_reasons"] = {r["reason"]: int(r["count"]) for r in
                           rejected.select(F.explode("reasons").alias("reason"))
                           .groupBy("reason").count().collect()}
    m["cleaned_values"] = ({r["col"]: int(r["count"]) for r in
                            accepted.select(F.explode("_cleaned_cols").alias("col"))
                            .groupBy("col").count().collect()}
                           if "_cleaned_cols" in accepted.columns else {})
    if spec.primary_key:
        m["pk_duplicates"] = (accepted.groupBy(*spec.primary_key).count()
                              .filter("count > 1").count())
    return m


def evaluate_table(spec: TableSpec, m: dict, enforce_expected_rows: bool = True) -> list[CheckResult]:
    t = spec.name
    out: list[CheckResult] = []
    add = lambda *a: out.append(CheckResult(t, *a))  # noqa: E731

    add("schema.required_columns", BLOCK, not m["missing_columns"],
        f"missing={m['missing_columns']}", "none missing")
    balance = m["rows_accepted"] + m["rows_rejected"]
    add("accounting.rows_in_equals_accepted_plus_rejected", BLOCK, balance == m["rows_in"],
        f"{m['rows_in']} in = {m['rows_accepted']} accepted + {m['rows_rejected']} rejected"
        f" (= {balance})", "exact equality")
    add("volume.non_empty", BLOCK, m["rows_accepted"] > 0, str(m["rows_accepted"]), "> 0")
    if spec.expected_rows:
        dev = abs(m["rows_in"] - spec.expected_rows) / spec.expected_rows
        add("volume.expected_rows", BLOCK if enforce_expected_rows else WARN,
            dev <= EXPECTED_ROWS_TOLERANCE, f"{m['rows_in']} ({dev:.2%} off {spec.expected_rows})",
            f"within {EXPECTED_ROWS_TOLERANCE:.0%}")
    rate = m["rows_rejected"] / max(m["rows_in"], 1)
    add("validity.reject_rate", BLOCK, rate <= spec.max_reject_rate,
        f"{rate:.4%} ({m['rows_rejected']} rows) {m['reject_reasons']}",
        f"<= {spec.max_reject_rate:.2%}")
    if spec.primary_key:
        add("uniqueness.primary_key", BLOCK, m["pk_duplicates"] == 0,
            f"{m['pk_duplicates']} duplicated keys", f"{'+'.join(spec.primary_key)} unique")
    if spec.name == "application_train":
        tr = m["target_rate"]
        lo, hi = TARGET_RATE_RANGE
        add("label.target_rate", BLOCK, tr is not None and lo <= tr <= hi,
            f"{tr:.4f}" if tr is not None else "n/a", f"[{lo}, {hi}]")
    high = {c: round(r, 3) for c, r in m["null_rate"].items() if r > HIGH_NULL_RATE}
    add("completeness.required_columns", WARN, not high, f"high null rate: {high}",
        f"<= {HIGH_NULL_RATE:.0%} null")
    return out


def evaluate_cross_table(train_ids: DataFrame, test_ids: DataFrame,
                         orphan_rates: dict[str, float]) -> list[CheckResult]:
    """Checks that need several tables: train/test overlap and referential integrity."""
    overlap = train_ids.join(test_ids, "SK_ID_CURR").count()
    out = [CheckResult("application_*", "cross.train_test_id_overlap", BLOCK, overlap == 0,
                       f"{overlap} shared SK_ID_CURR", "0")]
    for name, rate in orphan_rates.items():
        out.append(CheckResult(name.split(".")[0], f"integrity.orphans.{name.split('.', 1)[1]}", WARN,
                               rate <= MAX_ORPHAN_RATE, f"{rate:.2%} child rows without parent",
                               f"<= {MAX_ORPHAN_RATE:.0%}"))
    return out


def orphan_rate(child: DataFrame, child_col: str, parent_keys: DataFrame, parent_col: str) -> float:
    total = child.count()
    if total == 0:
        return 0.0
    parents = parent_keys.select(F.col(parent_col).alias("_pk")).distinct()
    orphans = child.join(parents, child[child_col] == parents["_pk"], "left_anti").count()
    return orphans / total


def raise_if_blocked(results: list[CheckResult]) -> None:
    failed = [r for r in results if r.severity == BLOCK and not r.passed]
    if failed:
        lines = "\n".join(f"  - {r.table_name}: {r.check} observed={r.observed} expected={r.threshold}"
                          for r in failed)
        raise QualityGateError(f"{len(failed)} blocking quality check(s) failed; "
                               f"silver tables were NOT published:\n{lines}")
