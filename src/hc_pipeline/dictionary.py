"""Column descriptions for Unity Catalog comments and the data dictionary.

Primary source: Kaggle's HomeCredit_columns_description.csv (part of the download,
latin-1 encoded). Columns added by our pipeline are described here.
"""
from __future__ import annotations

from .spec import TableSpec

DESCRIPTION_FILE = "HomeCredit_columns_description.csv"

PIPELINE_COLUMNS = {
    "DAYS_EMPLOYED_ANOM": "Added in silver: 1 if raw DAYS_EMPLOYED was the 365243 placeholder "
                          "(value set to NULL), else 0.",
    "_cleaned_cols": "Added in silver: columns of this row whose raw value was replaced by NULL "
                     "by a cleaning rule (placeholder 365243 or XNA/Unknown category).",
    "_source_file": "Lineage: raw file this row was read from.",
    "_source_version": "Lineage: raw data version folder (from ingestion).",
    "_silver_run_id": "Lineage: Databricks job run that produced this row.",
}

# Fallback descriptions for key columns if the Kaggle description file is unavailable.
FALLBACK = {
    "SK_ID_CURR": "ID of the loan application (applicant) in the sample.",
    "SK_ID_BUREAU": "ID of a Credit Bureau credit.",
    "SK_ID_PREV": "ID of a previous Home Credit credit/application.",
    "TARGET": "1 = client with payment difficulties (late > X days on one of the first Y "
              "installments), 0 = all other cases.",
    "MONTHS_BALANCE": "Month of the balance relative to the application date (-1 = most recent).",
    "STATUS": "Bureau credit status for the month: C closed, X unknown, 0 no DPD, 1-5 DPD buckets.",
    "DAYS_EMPLOYED": "Days before application the person started current employment "
                     "(365243 placeholder set to NULL in silver).",
    "DAYS_BIRTH": "Client's age in days at application (negative).",
}

_FILE_ALIASES = {"application_{train|test}.csv": ("application_train.csv", "application_test.csv")}


def load_descriptions(spark, raw_dir: str | None) -> dict[str, dict[str, str]]:
    """Return {file_name: {column: description}}; empty dict if the file is missing."""
    if not raw_dir:
        return {}
    try:
        df = (spark.read.option("header", True).option("encoding", "ISO-8859-1")
              .option("multiLine", True).option("escape", '"')
              .csv(f"{raw_dir}/{DESCRIPTION_FILE}"))
        rows = df.select("Table", "Row", "Description").collect()
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not read {DESCRIPTION_FILE}: {type(e).__name__}")
        return {}
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        if not r["Table"] or not r["Row"]:
            continue
        for f in _FILE_ALIASES.get(r["Table"], (r["Table"],)):
            out.setdefault(f, {})[r["Row"].strip()] = (r["Description"] or "").strip()
    return out


def column_comments(spec: TableSpec, descriptions: dict[str, dict[str, str]]) -> dict[str, str]:
    d = dict(FALLBACK)
    d.update(descriptions.get(spec.file, {}))
    d.update(PIPELINE_COLUMNS)
    return d
