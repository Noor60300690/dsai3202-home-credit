-- One-time Unity Catalog setup for the Home Credit project.
-- NOTE: our workspace already has the catalog `dbw_homecredit` (the workspace catalog) and
-- students cannot create catalogs, so the job uses dbw_homecredit (databricks.yml, var
-- `catalog`). Replace home_credit with dbw_homecredit below; skip CREATE CATALOG.
-- Run in a Databricks SQL editor or notebook as a user with CREATE CATALOG rights.
-- The job itself creates the schemas if they are missing (CREATE SCHEMA IF NOT EXISTS),
-- so after this file only the catalog, the external location and the grants are needed.

-- 1. Catalog ------------------------------------------------------------------------
-- If the metastore has no root storage, give the catalog a managed location inside
-- the project's ADLS Gen2 account (container `lakehouse`, created by the Azure setup).
CREATE CATALOG IF NOT EXISTS home_credit
  -- MANAGED LOCATION 'abfss://lakehouse@sahomecredit.dfs.core.windows.net/home_credit'
  COMMENT 'DSAI3202 Project: Home Credit Default Risk (Kaggle 2018). Loan default prediction.';

-- 2. Schemas (medallion layers + operations) -----------------------------------------
CREATE SCHEMA IF NOT EXISTS home_credit.bronze
  COMMENT 'Raw CSV rows as strings + ingestion metadata (_source_file, _source_sha256). One table per raw file.';
CREATE SCHEMA IF NOT EXISTS home_credit.silver_staging
  COMMENT 'Typed and cleaned tables waiting for the quality gate. Not for consumers.';
CREATE SCHEMA IF NOT EXISTS home_credit.silver
  COMMENT 'Published, typed, cleaned, de-duplicated tables. Only written after the quality gate passes.';
CREATE SCHEMA IF NOT EXISTS home_credit.gold
  COMMENT 'Model-ready outputs: features_v1 (one row per SK_ID_CURR), split_v1, feature_registry.';
CREATE SCHEMA IF NOT EXISTS home_credit.ops
  COMMENT 'Pipeline operations: run_log, table_metrics, quality_results, quarantine.';

-- 3. Access to the raw files (needed by bronze_load) --------------------------------
-- Uses the storage credential (access connector / managed identity) created during
-- the Azure setup.
-- CREATE EXTERNAL LOCATION IF NOT EXISTS home_credit_raw
--   URL 'abfss://raw@sahomecredit.dfs.core.windows.net/home_credit'
--   WITH (STORAGE CREDENTIAL <credential_name>)
--   COMMENT 'Versioned raw Kaggle files + manifest.json written by the ingestion job';
-- GRANT READ FILES ON EXTERNAL LOCATION home_credit_raw TO `<job_user_or_group>`;

-- 4. Grants for teammates (replace with your emails) --------------------------------
-- GRANT USE CATALOG, USE SCHEMA, SELECT, MODIFY, CREATE TABLE ON CATALOG home_credit TO `<teammate@domain>`;
-- GRANT USE CATALOG ON CATALOG home_credit TO `<instructor@domain>`;
-- GRANT USE SCHEMA, SELECT ON SCHEMA home_credit.silver TO `<instructor@domain>`;
-- GRANT USE SCHEMA, SELECT ON SCHEMA home_credit.gold   TO `<instructor@domain>`;

-- 5. Tags (optional, shows up in Catalog Explorer) ----------------------------------
ALTER CATALOG home_credit SET TAGS ('course' = 'DSAI3202', 'phase' = '1', 'source' = 'kaggle-home-credit-2018');
