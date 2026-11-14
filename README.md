# Predicting Loan Default with Borrowing-History Features

DSAI3202 Fall 2026, Project Phase 1: data pipeline, ETL and feature foundations.
Team: Lercia (ingestion, Azure setup, failure test, resource use) · Noor (ETL, quality gate,
catalog, EDA, features).

> Status markers: **TODO(Lercia)** / **TODO(Noor)** / **TODO(both)** = fill in after the Azure runs.

## Project and hypothesis

* **Problem:** estimate the probability that a loan applicant will have payment
  difficulties, to support approve / manual review / reject decisions for applicants with
  little credit history.
* **Target:** `TARGET` in `application_train` (1 = payment difficulties, ≈8 % positive).
* **Hypothesis:** adding features built from past borrowing history (bureau credits,
  previous applications, installment repayment behaviour) improves ROC-AUC by **≥ 0.03**
  over a model that uses only the application form.
* **Baseline plan:** logistic regression on the application group; comparison: LightGBM on
  application + history groups; control: LightGBM on the application group only. Same
  stratified 80/20 split (`gold.split_v1`) for all three.
* **Approved refinements:** TODO(both): none so far.

## Dataset and architecture

* **Source:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data),
  8 CSV tables + column descriptions, ≈2.5 GB uncompressed, ≈58 M rows.
* **Versions:** raw version folder `TODO(Lercia)`, manifest
  `raw/home_credit/<version>/manifest.json` (SHA-256 per file).
* **Schema:** tables, keys, types, cleaning rules: [docs/data_dictionary.md](docs/data_dictionary.md);
  every column profiled: [docs/data_dictionary_columns.md](docs/data_dictionary_columns.md).
* **Lineage:** [docs/lineage.md](docs/lineage.md).
* **Storage layout:**

  | Layer | Location | Content |
  |---|---|---|
  | raw | `abfss://raw@<acct>.dfs.core.windows.net/home_credit/<version>/` | original CSVs + manifest (never overwritten) |
  | bronze | `home_credit.bronze.*` | all-string Delta copy + `_source_*` metadata |
  | silver | `home_credit.silver.*` (staged in `silver_staging`) | typed, cleaned, de-duplicated |
  | gold | `home_credit.gold.features_v1`, `split_v1`, `feature_registry` | model inputs |
  | ops | `home_credit.ops.run_log`, `table_metrics`, `quality_results`, `quarantine` | evidence |

* **Architecture diagram:** TODO(both): diagram naming ADLS Gen2, Databricks (Jobs, Unity
  Catalog, Delta), Git repo and their roles.

## Run the pipeline

**Prerequisites:** Azure resource group with ADLS Gen2 + a Unity-Catalog-enabled
Databricks workspace (setup: TODO(Lercia)); the ingestion job has written a version
folder; Databricks CLI ≥ 0.230 authenticated (`databricks auth login`).

```bash
# 1. one-time catalog setup (SQL editor): sql/01_unity_catalog_setup.sql
# 2. set your workspace host + storage account in databricks.yml (variables raw_root, host)
pip install -r requirements-dev.txt
databricks bundle validate -t dev
databricks bundle deploy -t dev            # builds the wheel, creates the job
databricks bundle run -t dev home_credit_etl --params source_version=v1
```

Job `home_credit_etl` (single-node `Standard_D4ds_v5`, DBR 15.4 LTS):

`bronze_load → silver_transform → quality_gate → publish_silver → build_features_v1`

Each task logs to `ops.run_log`. `quality_gate` raises if any **BLOCK** check fails, so
`publish_silver` and `build_features_v1` don't run and the published silver tables stay
unchanged.

**Local (no Azure):** the same code runs on local Spark + Parquet:

```bash
pip install -r requirements-dev.txt
python scripts/make_synthetic_sample.py --out data/raw/synthetic_v1        # or use the real CSVs
PYTHONPATH=src python -m hc_pipeline.cli all --local --raw-root data/raw \
    --source-version synthetic_v1 --enforce-expected-rows false
pytest -q                                                                   # 11 unit tests
python scripts/build_data_dictionary.py --data-dir <folder with Kaggle CSVs>
python scripts/export_feature_spec.py
```

## Results and features

### ETL and quality

* **Run IDs:** successful run `TODO(Noor)`, rerun `TODO(Noor)`, controlled failure `TODO(Lercia)`.
* **Order of operations (silver):** trim + empty→NULL → cast with `try_cast` (failures =
  type violations) → business rules → exact duplicates → key conflicts → cleaning of
  accepted rows (365243 / XNA → NULL, `DAYS_EMPLOYED_ANOM` flag). Details:
  [data_dictionary.md §3-5](docs/data_dictionary.md).
* **Row accounting:** TODO(Noor): paste the output of the row-accounting query in
  `sql/02_verify_catalog_and_runs.sql` (rows in = accepted + rejected per table).
* **When publication stops (BLOCK checks):** missing required column; rows in ≠ accepted
  + rejected; empty table; row count > 1 % off the documented count (full data only);
  reject rate > 1 %; duplicate primary key in the accepted rows; train default rate outside
  5-12 %; train/test ID overlap. **WARN only:** columns > 50 % NULL, orphan child rows.
  Thresholds: `src/hc_pipeline/quality.py`, `spec.py`.
* **Safe rerun:** bronze is skipped when the version + SHA-256 match (`skipped_unchanged`);
  silver is overwritten (same input → same output); the quarantine is replaced per
  (table, source version), so a rerun doesn't duplicate it. Evidence: TODO(Noor).
* **Controlled failure:** `source_version=test_bad_v1` with `enforce_expected_rows=false`.
  Expected: `validity.reject_rate` fails on `application_train`, the job stops at
  `quality_gate`, and silver keeps the previous run's `hc.run_id`. Evidence: TODO(Lercia).
* **Catalog:** tables with comments (from Kaggle's description file), PK constraints and
  `hc.*` properties; see `sql/02_verify_catalog_and_runs.sql`. Screenshot: TODO(Noor).

### EDA

Notebook: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) (full population; one
documented 5 % systematic sample for a scatter plot). Figures: `docs/eda/`.
Findings: TODO(Noor): paste the summary cell output plus interpretation.

### Features

`gold.features_v1`: 50 features in two groups (24 application, 26 history), one row
per applicant. Definitions, sources, availability and leakage handling:
[docs/feature_spec.md](docs/feature_spec.md). Output version: Delta version `TODO(Noor)`,
`hc.git_commit` = `TODO`.

## Resource use and contributions

* **Cost / performance choice:** TODO(Lercia): CSV vs Delta measurement.
* **Cleanup:** TODO(Lercia).
* **Contributions:**
  * Lercia: TODO
  * Noor: ETL bronze→silver with quarantine; quality gate, row accounting and Databricks Job
    (bundle); data dictionary, lineage and Unity Catalog registration; EDA notebook;
    feature set v1, split and feature spec; unit tests.
* **Reused code:** TODO(both): cite lab code adapted (Lab-3…7) and what was changed.

## AI use

| Tool | Request | Response | Our changes / checks |
|---|---|---|---|
| Claude (Anthropic) | Dataset ranking and hypothesis drafting (Project 0) | see Project 0 submission | see Project 0 submission |
| Claude (Anthropic), used by Noor | Build Noor's Phase-1 part from the project brief and our hypothesis: ETL bronze→silver with quarantine, quality gate + Databricks Job, data dictionary/lineage/Unity Catalog registration, EDA notebook, feature set v1 with leakage notes | Python package `hc_pipeline` (ETL, quality gate, features), Databricks bundle (`databricks.yml`), SQL, docs, EDA notebook, 12 unit tests and a synthetic test-data generator. Claude also drafted the EDA interpretation text and the commit messages. | **Run and checked on the real data by Noor:** ran the profiler on all 8 raw files (58.5 M rows) and confirmed row counts, keys and duplicates against the hypothesis document; investigated the single type failure (`DAYS_REGISTRATION` = −10116.04 for applicant 408583) and chose to keep the applicant (typed as DOUBLE) instead of quarantining it; ran the EDA notebook on the full data and checked the findings that the interpretation is based on. **Adapted to our Azure environment:** found that students can't create catalogs/schemas or use the access connector (blocked by the subscription policy), so the job was changed to use the existing catalog `dbw_homecredit` and a single schema (`default`) with prefixed tables, reading raw files from a Unity Catalog Volume; deployed and ran the job on Databricks. Set up the GitHub repo and coordinated the storage layout and manifest format with Lercia. |
