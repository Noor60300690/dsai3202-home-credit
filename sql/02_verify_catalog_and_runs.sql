-- Evidence queries for the README. Run after a job run; screenshot or export the results.

-- Catalog registration: tables, comments, keys
SHOW TABLES IN home_credit.silver;
DESCRIBE TABLE EXTENDED home_credit.silver.application_train;   -- comments, PK constraint, properties
SHOW TBLPROPERTIES home_credit.silver.application_train;
SELECT table_name, comment FROM home_credit.information_schema.tables WHERE table_schema = 'silver';
SELECT table_name, column_name, data_type, comment
FROM home_credit.information_schema.columns
WHERE table_schema = 'silver' AND table_name = 'application_train'
ORDER BY ordinal_position;
SELECT * FROM home_credit.information_schema.table_constraints WHERE table_schema = 'silver';

-- Row accounting for one run (replace the run id)
SELECT table_name,
       get_json_object(metrics_json, '$.rows_in')        AS rows_in,
       get_json_object(metrics_json, '$.rows_accepted')  AS rows_accepted,
       get_json_object(metrics_json, '$.rows_rejected')  AS rows_rejected,
       get_json_object(metrics_json, '$.reject_reasons') AS reject_reasons,
       get_json_object(metrics_json, '$.cleaned_values') AS cleaned_values
FROM home_credit.ops.table_metrics
WHERE run_id = '<run_id>'
ORDER BY table_name;

-- Quality gate results for a run (successful run and the controlled failure)
SELECT run_id, source_version, table_name, check_name, severity, passed, observed, threshold
FROM home_credit.ops.quality_results
WHERE run_id = '<run_id>'
ORDER BY passed, severity, table_name;

-- Safe rerun evidence: bronze skipped, same silver counts, quarantine not duplicated
SELECT run_id, task, table_name, status, rows, detail, logged_at
FROM home_credit.ops.run_log ORDER BY logged_at DESC LIMIT 100;
SELECT table_name, source_version, count(*) AS rejected_rows, count(DISTINCT run_id) AS runs
FROM home_credit.ops.quarantine GROUP BY ALL;

-- Delta history = output versions
DESCRIBE HISTORY home_credit.silver.application_train;
DESCRIBE HISTORY home_credit.gold.features_v1;
