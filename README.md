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
  | raw | `abfss://raw@sahomecredit.dfs.core.windows.net/home_credit/<version>/` | original CSVs + manifest (never overwritten) |
  | bronze | `dbw_homecredit.default.bronze_<table>` | all-string Delta copy + `_source_*` metadata |
  | silver | `dbw_homecredit.default.silver_<table>` (staged in `silver_staging_<table>`) | typed, cleaned, de-duplicated |
  | gold | `dbw_homecredit.default.gold_features_v1`, `gold_split_v1`, `gold_feature_registry` | model inputs |
  | ops | `dbw_homecredit.default.ops_run_log`, `ops_table_metrics`, `ops_quality_results`, `ops_quarantine` | evidence |

  **Why one schema with prefixes:** in our course workspace students have `USE CATALOG` on
  the workspace catalog `dbw_homecredit` but cannot create catalogs or schemas; we can
  create tables in `dbw_homecredit.default`. The code supports both layouts (job parameter
  `single_schema`; empty = separate `bronze`/`silver`/`gold`/`ops` schemas).

* **Architecture diagram:** TODO(both): diagram naming ADLS Gen2, Databricks (Jobs, Unity
  Catalog, Delta), Git repo and their roles.

## Run the pipeline

**Prerequisites:** Azure resource group with ADLS Gen2 + a Unity-Catalog-enabled
Databricks workspace (setup: TODO(Lercia)); the ingestion job has written a version
folder; Databricks CLI ≥ 0.230 (`brew install databricks/tap/databricks`).

**Storage access:** creating an Access Connector / storage credential is blocked by the
course subscription policy, so the job cluster reads ADLS with the storage account key,
kept in a Databricks secret scope (never in Git). One-time setup:

```bash
databricks auth login --host https://adb-7405604671478132.12.azuredatabricks.net
databricks secrets create-scope hc
read -s "KEY?Paste the key1 value of sahomecredit (Access keys): "
databricks secrets put-secret hc sahomecredit-key --string-value "$KEY" && unset KEY
```

**Deploy and run** (workspace host, storage account and catalog are already set in
`databricks.yml`):

```bash
pip install -r requirements-dev.txt
databricks bundle validate -t dev
databricks bundle deploy -t dev            # builds the wheel, creates the job
databricks bundle run -t dev home_credit_etl                     # raw version v1 (default)
# other inputs, e.g. the controlled failure test:
databricks bundle run -t dev home_credit_etl --params source_version=test_bad_v1,enforce_expected_rows=false
```

Job `home_credit_etl` (single-node `Standard_D4ds_v5`, DBR 15.4 LTS):

`bronze_load → silver_transform → quality_gate → publish_silver → build_features_v1`

Each task logs to `ops_run_log`. `quality_gate` raises if any **BLOCK** check fails, so
`publish_silver` and `build_features_v1` don't run and the published silver tables stay
unchanged.

**Local (no Azure):** the same code runs on local Spark + Parquet:

```bash
pip install -r requirements-dev.txt
python scripts/make_synthetic_sample.py --out data/raw/synthetic_v1        # or use the real CSVs
PYTHONPATH=src python -m hc_pipeline.cli all --local --raw-root data/raw \
    --source-version synthetic_v1 --enforce-expected-rows false
pytest -q                                                                   # 12 unit tests
python scripts/build_data_dictionary.py --data-dir <folder with Kaggle CSVs>
python scripts/export_feature_spec.py
```

## Results and features

### ETL and quality

* **Run IDs** (job `68730613515617`, workspace `adb-7405604671478132`): successful full run
  `933468886966780` (2026-11-14, 20:34 → 21:15, raw version `v1`), rerun `TODO(Noor)`,
  controlled failure `TODO(Lercia)`. Earlier runs: `1073714360410313` = rehearsal on synthetic
  data (passed); `1109123459055160` and `601695391609100` = failed in `bronze_load` because the
  storage key secret was saved incorrectly (fixed, see run instructions).
* **Order of operations (silver):** trim + empty→NULL → cast with `try_cast` (failures =
  type violations) → business rules → exact duplicates → key conflicts → cleaning of
  accepted rows (365243 / XNA → NULL, `DAYS_EMPLOYED_ANOM` flag). Details:
  [data_dictionary.md §3-5](docs/data_dictionary.md).
* **Row accounting** (run `933468886966780`, full data; rows in = accepted + rejected for every table):

  | Table | Rows in (bronze) | Accepted (silver) | Rejected (quarantine) | Values cleaned (→ NULL) |
  |---|---:|---:|---:|---|
  | application_train | 307,511 | 307,511 | 0 | DAYS_EMPLOYED 55,374 · ORGANIZATION_TYPE 55,374 · CODE_GENDER 4 · NAME_FAMILY_STATUS 2 |
  | application_test | 48,744 | 48,744 | 0 | DAYS_EMPLOYED 9,274 · ORGANIZATION_TYPE 9,274 |
  | bureau | 1,716,428 | 1,716,428 | 0 | – |
  | bureau_balance | 27,299,925 | 27,299,925 | 0 | – |
  | previous_application | 1,670,214 | 1,670,214 | 0 | NAME_CASH_LOAN_PURPOSE 1,600,579 · DAYS_FIRST_DRAWING 934,444 · DAYS_TERMINATION 225,913 · DAYS_LAST_DUE 211,221 · DAYS_LAST_DUE_1ST_VERSION 93,864 · DAYS_FIRST_DUE 40,645 · CODE_REJECT_REASON 5,244 |
  | installments_payments | 13,605,401 | 13,605,401 | 0 | – |
  | pos_cash_balance | 10,001,358 | 10,001,358 | 0 | – |
  | credit_card_balance | 3,840,312 | 3,840,312 | 0 | – |
  | **Total** | **58,489,893** | **58,489,893** | **0** | |

  Quality gate: 58 checks, **0 blocking failures**; all row counts match the documented
  Kaggle counts exactly; train default rate 8.07 %; 0 train/test ID overlap.
  Warnings: `EXT_SOURCE_1` is 56.4 % NULL (expected, kept as NULL); 11.43 % of
  `bureau_balance` rows reference a `SK_ID_BUREAU` that is not in `bureau` (orphans, kept in
  silver; they cannot reach an applicant, so the feature joins drop them).
  Observation: the 55,374 `ORGANIZATION_TYPE = 'XNA'` rows are exactly the 55,374
  `DAYS_EMPLOYED = 365243` rows, i.e. applicants without an employer (mostly pensioners),
  which supports keeping `DAYS_EMPLOYED_ANOM` as a feature.
  Zero rejected rows is the expected result on this source: the profiler found one type
  failure (`DAYS_REGISTRATION`, applicant 408583), which was resolved in the schema
  contract, and no duplicates. The quarantine path is shown by the synthetic rehearsal
  (3 duplicate `bureau` rows quarantined) and by the controlled failure test.
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
  `hc.*` properties; see `sql/02_verify_catalog_and_runs.sql`.
  Evidence: [successful job run](docs/evidence/01_job_run_success.png) (5 tasks green, 40 min
  on single-node Standard_D4ds_v5: bronze 11m40s, silver 13m12s, gate 30s, publish 13m33s,
  features 1m19s) and [Unity Catalog registration](docs/evidence/02_catalog_registration.png)
  (31 tables in `dbw_homecredit.default`, table description, `SK_ID_CURR` primary key,
  column comments from Kaggle's description file). [Lineage](docs/evidence/03_lineage.png)
  recorded automatically by Unity Catalog: `silver_staging_application_train` → (job) →
  `silver_application_train` → (job) → `gold_features_v1`.

### EDA

Notebook: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) (full population; one
documented 5 % systematic sample for a scatter plot). Figures: `docs/eda/`.
Findings (full data, 58.5 M rows):

* **Target imbalance:** 8.07 % defaults → ROC-AUC and recall at a fixed approval rate, stratified split; never accuracy.
* **Missing values:** 71 of 121 application columns have NULLs, 41 are > 50 % empty (EXT_SOURCE_1: 56 %) → kept as NULL in silver, imputed in training only.
* **DAYS_EMPLOYED placeholder:** 55,374 rows (default rate 5.40 % vs 8.66 %) → NULL + `DAYS_EMPLOYED_ANOM` flag, because the placeholder itself is informative.
* **Outliers:** max income is 248× its 99th percentile (159 extreme rows) → ratio features / log scale; tree models are robust.
* **Strongest single signals:** EXT_SOURCE_3 (−0.179), EXT_SOURCE_2 (−0.160), EXT_SOURCE_1 (−0.155), DAYS_BIRTH (+0.078).
* **Collinear pairs (|r| > 0.8):** AMT_CREDIT & AMT_GOODS_PRICE, CNT_CHILDREN & CNT_FAM_MEMBERS → matters for the logistic baseline.
* **History coverage:** 86 % have bureau records, 95 % previous applications; 1 % have neither.
* **Hypothesis signal:** default rate is 6.8 % for applicants never late on past installments vs 11.9 % for those late on > 20 % of installments.

Interpretation: see the last cell of the notebook.

### Features

`gold_features_v1`: 50 features in two groups (24 application, 26 history), one row
per applicant: **356,255 rows** (307,511 train + 48,744 test applications).
`gold_split_v1` (stratified, seed 3202): train 246,009 (19,860 defaults) / test 61,502
(4,965 defaults), default rate 8.07 % in both. Definitions, sources, availability and leakage handling:
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
| Claude (Anthropic), used by Noor | Help implement Noor's Phase-1 part from the brief and our hypothesis: ETL bronze→silver with quarantine, quality gate + Databricks Job, data dictionary/lineage/Unity Catalog, EDA notebook, feature set v1 with leakage notes | Generated the first versions of the Python package (`hc_pipeline`), the Databricks bundle (`databricks.yml`), SQL, docs, the EDA notebook text, unit tests and a synthetic test-data generator; helped debug errors during the Azure runs | **Validation on real data:** Noor ran the profiler on all 8 raw files (58.5 M rows), confirmed row counts, keys and duplicates against our hypothesis document, and found the one type failure (`DAYS_REGISTRATION` = −10116.04, applicant 408583); decided to keep that applicant (typed as DOUBLE) instead of quarantining. Ran the EDA on the full data and checked the findings. **Azure adaptation and debugging:** discovered that students cannot create catalogs/schemas and that the access connector is blocked; adapted the job to the existing catalog `dbw_homecredit` (single schema `default`, prefixed tables) and to reading ADLS with the storage key from a Databricks secret scope; set up the CLI, deployed the bundle, ran the test run on synthetic data and the full run on the real data, and fixed the secret configuration. **Understanding:** walked through the design (schema contract and rules, quarantine, BLOCK vs WARN quality checks, leakage filter on history dates) and can explain each component. **Team:** created the GitHub repo and agreed the raw layout and manifest format with Lercia. |
