"""Entry points for the Databricks Job tasks (also runnable locally with --local).

    bronze   raw CSV (versioned folder + manifest.json)  -> <catalog>.bronze.*
    silver   bronze -> silver_staging.* + ops.quarantine + ops.table_metrics
    gate     ops.table_metrics -> ops.quality_results; raises if a BLOCK check fails
    publish  silver_staging.* -> silver.* (+ comments, PK, table properties)
    features silver.* -> gold.features_v1, gold.split_v1, gold.feature_registry
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from . import dictionary, features, quality, transforms
from .spec import TABLES
from .store import Store, make_store


# --------------------------------------------------------------------------- helpers
def _args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["bronze", "silver", "gate", "publish", "features", "all"])
    p.add_argument("--catalog", default="home_credit")
    p.add_argument("--raw-root", required=False, default="",
                   help="Folder that holds one sub-folder per source version.")
    p.add_argument("--source-version", required=True,
                   help="Raw data version folder, e.g. v1 or test_bad_v1.")
    p.add_argument("--run-id", default=None, help="Defaults to a UTC timestamp.")
    p.add_argument("--git-commit", default="unknown")
    p.add_argument("--tables", default=",".join(TABLES), help="Comma-separated subset.")
    p.add_argument("--enforce-expected-rows", default="true",
                   help="true for the full Kaggle data; false for samples/test inputs.")
    p.add_argument("--force", action="store_true", help="Reload bronze even if unchanged.")
    p.add_argument("--local", action="store_true", help="Local Spark + Parquet (no Azure).")
    a = p.parse_args(argv)
    a.run_id = a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    a.tables = [t for t in a.tables.split(",") if t]
    a.enforce_expected_rows = str(a.enforce_expected_rows).lower() == "true"
    return a


def _spark(local: bool) -> SparkSession:
    if not local:
        return SparkSession.builder.getOrCreate()
    return (SparkSession.builder.master("local[2]").appName("hc-pipeline-local")
            .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "4")
            .config("spark.sql.warehouse.dir", "spark-warehouse")
            .enableHiveSupport().getOrCreate())


def _log(store: Store, a, task: str, table: str, status: str, rows: int | None = None,
         detail: str = "") -> None:
    print(f"[{task}] {table}: {status} rows={rows} {detail}")
    row = [(a.run_id, task, table, status, rows, a.source_version, a.git_commit, detail,
            datetime.now(timezone.utc))]
    schema = ("run_id string, task string, table_name string, status string, rows bigint, "
              "source_version string, git_commit string, detail string, logged_at timestamp")
    store.append(store.spark.createDataFrame(row, schema), "ops", "run_log")


def _raw_dir(a) -> str:
    return f"{a.raw_root.rstrip('/')}/{a.source_version}"


def read_manifest(spark: SparkSession, raw_dir: str) -> dict[str, str]:
    """Map file name -> sha256 from the ingestion manifest. Tolerates the common shapes
    {"files": [{"name"|"file"|"path": ..., "sha256"|"hash": ...}]} or {"<file>": "<sha>"}."""
    try:
        text = spark.read.text(f"{raw_dir}/manifest.json", wholetext=True).first()[0]
    except Exception:  # noqa: BLE001
        print(f"WARNING: no manifest.json in {raw_dir}; bronze will carry no file hash")
        return {}
    data = json.loads(text)
    entries = data.get("files", data) if isinstance(data, dict) else data
    if isinstance(entries, dict):
        return {k.split("/")[-1]: (v if isinstance(v, str) else v.get("sha256", v.get("hash")))
                for k, v in entries.items()}
    out = {}
    for e in entries:
        name = e.get("name") or e.get("file") or e.get("path") or e.get("file_name")
        out[str(name).split("/")[-1]] = e.get("sha256") or e.get("hash") or e.get("md5")
    return out


# --------------------------------------------------------------------------- tasks
def task_bronze(spark, store: Store, a) -> None:
    raw_dir = _raw_dir(a)
    manifest = read_manifest(spark, raw_dir)
    for name in a.tables:
        spec = TABLES[name]
        sha = manifest.get(spec.file)
        prev = store.props("bronze", name)
        if (not a.force and prev.get("hc.source_version") == a.source_version
                and prev.get("hc.source_sha256") == str(sha)):
            _log(store, a, "bronze", name, "skipped_unchanged", None,
                 f"already loaded from {a.source_version} sha256={sha}")
            continue
        raw = transforms.read_raw_csv(spark, f"{raw_dir}/{spec.file}")
        bronze = transforms.to_bronze(raw, a.source_version, sha, a.run_id)
        store.overwrite(bronze, "bronze", name, {
            "hc.source_version": a.source_version, "hc.source_sha256": str(sha),
            "hc.source_path": f"{raw_dir}/{spec.file}", "hc.run_id": a.run_id})
        _log(store, a, "bronze", name, "written", store.read("bronze", name).count())


def task_silver(spark, store: Store, a) -> None:
    for name in a.tables:
        spec = TABLES[name]
        bronze = store.read("bronze", name).filter(F.col("_source_version") == a.source_version)
        bronze_rows = bronze.count()
        out = transforms.bronze_to_silver(bronze, spec, a.run_id)
        store.overwrite(out["accepted"], "silver_staging", name,
                        {"hc.source_version": a.source_version, "hc.run_id": a.run_id})
        store.replace_partition(out["rejected"], "ops", "quarantine",
                                f"table_name = '{name}' AND source_version = '{a.source_version}'")
        staged = store.read("silver_staging", name)
        rejected = store.read("ops", "quarantine").filter(
            (F.col("table_name") == name) & (F.col("run_id") == a.run_id))
        m = quality.table_metrics(spec, bronze_rows, staged, rejected, out["missing_columns"])
        store.append(spark.createDataFrame(
            [(a.run_id, name, a.source_version, json.dumps(m), datetime.now(timezone.utc))],
            "run_id string, table_name string, source_version string, metrics_json string, "
            "measured_at timestamp"), "ops", "table_metrics")
        _log(store, a, "silver", name, "staged", m["rows_accepted"],
             f"in={m['rows_in']} rejected={m['rows_rejected']} reasons={m['reject_reasons']} "
             f"cleaned={m['cleaned_values']}")


def task_gate(spark, store: Store, a) -> None:
    rows = (store.read("ops", "table_metrics").filter(F.col("run_id") == a.run_id).collect())
    metrics = {r["table_name"]: json.loads(r["metrics_json"]) for r in rows}
    results: list[quality.CheckResult] = []
    for name in a.tables:
        if name not in metrics:
            results.append(quality.CheckResult(name, "run.metrics_present", quality.BLOCK, False,
                                               "no metrics for this run", "silver task ran"))
            continue
        results += quality.evaluate_table(TABLES[name], metrics[name], a.enforce_expected_rows)

    staged = {n: store.read("silver_staging", n) for n in a.tables if store.exists("silver_staging", n)}
    if {"application_train", "application_test"} <= staged.keys():
        tr, te = (staged[n].select("SK_ID_CURR") for n in ("application_train", "application_test"))
        app_ids = tr.unionByName(te)
        orphan = {}
        for child in ("bureau", "previous_application"):
            if child in staged:
                orphan[f"{child}.SK_ID_CURR"] = quality.orphan_rate(staged[child], "SK_ID_CURR",
                                                                    app_ids, "SK_ID_CURR")
        if {"bureau_balance", "bureau"} <= staged.keys():
            orphan["bureau_balance.SK_ID_BUREAU"] = quality.orphan_rate(
                staged["bureau_balance"], "SK_ID_BUREAU", staged["bureau"], "SK_ID_BUREAU")
        results += quality.evaluate_cross_table(tr, te, orphan)

    out = spark.createDataFrame(
        [(a.run_id, a.source_version, r.table_name, r.check, r.severity, r.passed, r.observed,
          r.threshold, datetime.now(timezone.utc)) for r in results],
        "run_id string, source_version string, table_name string, check_name string, "
        "severity string, passed boolean, observed string, threshold string, checked_at timestamp")
    store.append(out, "ops", "quality_results")
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'} [{r.severity}] {r.table_name:24s} {r.check}: "
              f"{r.observed} (expected {r.threshold})")
    failed = [r for r in results if r.severity == quality.BLOCK and not r.passed]
    _log(store, a, "gate", "*", "failed" if failed else "passed", len(results),
         f"{len(failed)} blocking failures")
    quality.raise_if_blocked(results)


def task_publish(spark, store: Store, a) -> None:
    passed = (store.read("ops", "run_log")
              .filter((F.col("run_id") == a.run_id) & (F.col("task") == "gate")
                      & (F.col("status") == "passed")).count())
    if not passed:
        raise quality.QualityGateError(f"No passed quality gate for run {a.run_id}; refusing to publish.")
    descriptions = dictionary.load_descriptions(spark, _raw_dir(a) if a.raw_root else None)
    for name in a.tables:
        spec = TABLES[name]
        staged = store.read("silver_staging", name)
        store.overwrite(staged, "silver", name, {
            "hc.source_version": a.source_version, "hc.run_id": a.run_id,
            "hc.git_commit": a.git_commit, "hc.quality_gate": "passed",
            "hc.lineage": f"raw/{a.source_version}/{spec.file} -> bronze.{name} -> silver.{name}"})
        warns = store.apply_metadata("silver", name, spec.description,
                                     dictionary.column_comments(spec, descriptions), spec.primary_key)
        _log(store, a, "publish", name, "published", store.read("silver", name).count(),
             "; ".join(warns) + f" delta_version={store.current_version('silver', name)}")


def task_features(spark, store: Store, a) -> None:
    needed = ["application_train", "application_test", "bureau", "bureau_balance",
              "previous_application", "installments_payments", "pos_cash_balance",
              "credit_card_balance"]
    silver = {n: store.read("silver", n) for n in needed}
    versions = {n: store.current_version("silver", n) for n in needed}
    feats = features.build_features(silver)
    props = {"hc.feature_version": features.FEATURE_VERSION, "hc.source_version": a.source_version,
             "hc.run_id": a.run_id, "hc.git_commit": a.git_commit,
             "hc.silver_versions": json.dumps(versions).replace("'", "")}
    store.overwrite(feats, "gold", f"features_{features.FEATURE_VERSION}", props)
    split = features.stratified_split(store.read("gold", f"features_{features.FEATURE_VERSION}"))
    store.overwrite(split, "gold", f"split_{features.FEATURE_VERSION}",
                    {**props, "hc.split_seed": features.SPLIT_SEED,
                     "hc.test_fraction": features.TEST_FRACTION})
    registry = spark.createDataFrame(
        [(f.name, f.group, f.dtype, f.source, f.definition, f.leakage_note, features.FEATURE_VERSION)
         for f in features.FEATURES],
        "feature string, feature_group string, dtype string, source string, definition string, "
        "leakage_note string, feature_version string")
    store.overwrite(registry, "gold", "feature_registry", props)
    n = store.read("gold", f"features_{features.FEATURE_VERSION}").count()
    s = store.read("gold", f"split_{features.FEATURE_VERSION}").groupBy("SPLIT", "TARGET").count().collect()
    _log(store, a, "features", f"features_{features.FEATURE_VERSION}", "written", n,
         "split=" + ", ".join(f"{r['SPLIT']}/{r['TARGET']}:{r['count']}" for r in sorted(s, key=str)))


TASKS = {"bronze": task_bronze, "silver": task_silver, "gate": task_gate,
         "publish": task_publish, "features": task_features}


def main(argv=None) -> None:
    a = _args(argv)
    spark = _spark(a.local)
    store = make_store(spark, a.catalog, a.local)
    store.ensure_schemas()
    for t in (["bronze", "silver", "gate", "publish", "features"] if a.task == "all" else [a.task]):
        TASKS[t](spark, store, a)


# Wheel entry points (one per Databricks task) ------------------------------------
def _entry(task: str):
    def run():
        main([task, *sys.argv[1:]])
    return run


bronze, silver, gate, publish, build_features = (_entry(t) for t in
                                                 ("bronze", "silver", "gate", "publish", "features"))

if __name__ == "__main__":
    main()
