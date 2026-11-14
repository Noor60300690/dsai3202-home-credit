-- Evidence queries for the README (single-schema layout: dbw_homecredit.default.<layer>_<table>). Run after a job run; screenshot or export the results.

-- Catalog registration: tables, comments, keys
SHOW TABLES IN dbw_homecredit.default;
DESCRIBE TABLE EXTENDED dbw_homecredit.default.silver_application_train;   -- comments, PK constraint, properties
SHOW TBLPROPERTIES dbw_homecredit.default.silver_application_train;
SELECT table_name, comment FROM dbw_homecredit.information_schema.tables WHERE table_schema = 'default' AND table_name LIKE 'silver_%';
SELECT table_name, column_name, data_type, comment
FROM dbw_homecredit.information_schema.columns
WHERE table_schema = 'default' AND table_name = 'silver_application_train'
ORDER BY ordinal_position;
SELECT * FROM dbw_homecredit.information_schema.table_constraints WHERE table_schema = 'default' AND table_name LIKE 'silver_%';

-- Row accounting for one run (replace the run id)
SELECT table_name,
       get_json_object(metrics_json, '$.rows_in')        AS rows_in,
       get_json_object(metrics_json, '$.rows_accepted')  AS rows_accepted,
       get_json_object(metrics_json, '$.rows_rejected')  AS rows_rejected,
       get_json_object(metrics_json, '$.reject_reasons') AS reject_reasons,
       get_json_object(metrics_json, '$.cleaned_values') AS cleaned_values
FROM dbw_homecredit.default.ops_table_metrics
WHERE run_id = '<run_id>'
ORDER BY table_name;

-- Quality gate results for a run (successful run and the controlled failure)
SELECT run_id, source_version, table_name, check_name, severity, passed, observed, threshold
FROM dbw_homecredit.default.ops_quality_results
WHERE run_id = '<run_id>'
ORDER BY passed, severity, table_name;

-- Safe rerun evidence: bronze skipped, same silver counts, quarantine not duplicated
SELECT run_id, task, table_name, status, rows, detail, logged_at
FROM dbw_homecredit.default.ops_run_log ORDER BY logged_at DESC LIMIT 100;
SELECT table_name, source_version, count(*) AS rejected_rows, count(DISTINCT run_id) AS runs
FROM dbw_homecredit.default.ops_quarantine GROUP BY ALL;

-- Delta history = output versions
DESCRIBE HISTORY dbw_homecredit.default.silver_application_train;
DESCRIBE HISTORY dbw_homecredit.default.gold_features_v1;
