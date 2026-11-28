# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2 — model training, validation and final test evaluation (Noor)
# MAGIC
# MAGIC Brief II.1 and II.2. Reads `gold_features_v1` and `gold_split_v1` from Phase 1 and:
# MAGIC
# MAGIC 1. carves a **validation** set out of the Phase-1 train split (same hash rule, seed 3203);
# MAGIC    the Phase-1 **test** set (61,502 rows) is not used until section 5;
# MAGIC 2. trains the **baseline** (logistic regression, application form only), the **control**
# MAGIC    (LightGBM, application form only) and the **candidate** (LightGBM, application form +
# MAGIC    borrowing history), with a small LightGBM grid judged on validation only;
# MAGIC 3. records the feature-table Delta version, git commit, library versions, parameters,
# MAGIC    seeds and MLflow run IDs;
# MAGIC 4. scores every model **once** on the test set (ROC-AUC, PR-AUC, Brier), tests the
# MAGIC    hypothesis with a paired bootstrap, and repeats the comparison without `EXT_SOURCE_*`;
# MAGIC 5. error analysis: confident mistakes and the weakest slice;
# MAGIC 6. exports what deployment needs: the model, its tags, the input contract, a fixed data
# MAGIC    snapshot and 100 fixed test rows with expected outputs for the consistency check.
# MAGIC
# MAGIC **Run it on** a single-node cluster with DBR 15.4 LTS (or 15.4 LTS ML), from the repo's
# MAGIC Git folder so that `../src` is importable and evidence is written to `docs/evidence/phase2/`.
# MAGIC Fill in the `git_commit` widget with the commit you are running (`git rev-parse HEAD`).

# COMMAND ----------

# MAGIC %pip install lightgbm==4.5.0

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821  (Databricks global)

# COMMAND ----------

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("../src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from hc_pipeline import features, modeling
from hc_pipeline.store import make_store

spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("catalog", "dbw_homecredit")                     # noqa: F821
dbutils.widgets.text("single_schema", "default")                      # noqa: F821
dbutils.widgets.text("git_commit", "")                                # noqa: F821
dbutils.widgets.text("experiment", "")                                # noqa: F821
dbutils.widgets.text("evidence_dir", "../docs/evidence/phase2")       # noqa: F821
dbutils.widgets.text("export_dir", "/Volumes/dbw_homecredit/default/raw/phase2")  # noqa: F821
dbutils.widgets.dropdown("expect_full_data", "true", ["true", "false"])  # noqa: F821
dbutils.widgets.dropdown("local", "false", ["true", "false"])        # noqa: F821
W = {k: dbutils.widgets.get(k) for k in (                              # noqa: F821
    "catalog", "single_schema", "git_commit", "experiment", "evidence_dir", "export_dir",
    "expect_full_data", "local")}

if not W["git_commit"].strip():
    raise ValueError("Set the git_commit widget to the commit you are running "
                     "(`git rev-parse HEAD`, or the commit shown in the Git folder dialog).")
GIT_COMMIT = W["git_commit"].strip()
EVIDENCE = os.path.abspath(W["evidence_dir"])
os.makedirs(EVIDENCE, exist_ok=True)
store = make_store(spark, W["catalog"], local=W["local"] == "true", single_schema=W["single_schema"])
print("evidence ->", EVIDENCE)

# COMMAND ----------

# MAGIC %md ## 1. Inputs: check they are the real v1 data, record their versions

# COMMAND ----------

FEAT_TABLE = f"features_{features.FEATURE_VERSION}"
feat_sdf = store.read("gold", FEAT_TABLE)
split_sdf = store.read("gold", f"split_{features.FEATURE_VERSION}")
feat_props = store.props("gold", FEAT_TABLE)

checks = {
    "feature_source_version_is_v1": feat_props.get("hc.source_version") == "v1",
    "columns_match_spec": feat_sdf.columns == features.ID_COLUMNS + features.FEATURE_NAMES,
}
if W["expect_full_data"] == "true":
    checks["feature_rows_356255"] = feat_sdf.count() == 356_255
    checks["test_rows_61502"] = split_sdf.filter("SPLIT = 'test'").count() == 61_502
    checks["train_rows_246009"] = split_sdf.filter("SPLIT = 'train'").count() == 246_009
print(json.dumps(checks, indent=2))
assert all(checks.values()), f"Input check failed: {checks}. Rerun the Phase-1 v1 job first."

# Train / validation / test. Test rows are passed through unchanged from gold_split_v1.
tvt_sdf = features.validation_split(split_sdf)
TVT_TABLE = f"split_tvt_{features.FEATURE_VERSION}"
store.overwrite(tvt_sdf, "gold", TVT_TABLE, {
    "hc.feature_version": features.FEATURE_VERSION, "hc.git_commit": GIT_COMMIT,
    "hc.split_seed": features.SPLIT_SEED, "hc.validation_seed": features.VALIDATION_SEED,
    "hc.validation_fraction": features.VALIDATION_FRACTION,
    "hc.source_split_table": store.fq("gold", f"split_{features.FEATURE_VERSION}")})

versions = {
    "features": store.current_version("gold", FEAT_TABLE),
    "split": store.current_version("gold", f"split_{features.FEATURE_VERSION}"),
    "split_tvt": store.current_version("gold", TVT_TABLE),
}
print("Delta versions:", versions)

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
df = (feat_sdf.filter("DATASET = 'train'")
      .join(store.read("gold", TVT_TABLE).select("SK_ID_CURR", "SPLIT"), "SK_ID_CURR")
      .toPandas().sort_values("SK_ID_CURR").reset_index(drop=True))
df["TARGET"] = df["TARGET"].astype(int)
split_counts = {f"{s}/{t}": int(n) for (s, t), n in df.groupby(["SPLIT", "TARGET"]).size().items()}
print(split_counts)

parts = {s: df[df["SPLIT"] == s].reset_index(drop=True) for s in ("train", "validation", "test")}
X = {s: modeling.to_model_frame(d) for s, d in parts.items()}
y = {s: d["TARGET"].to_numpy() for s, d in parts.items()}
assert set(parts["test"]["SK_ID_CURR"]).isdisjoint(parts["train"]["SK_ID_CURR"])
assert set(parts["validation"]["SK_ID_CURR"]).isdisjoint(parts["train"]["SK_ID_CURR"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train and tune (validation set only)
# MAGIC
# MAGIC * Logistic regression: median imputation (+ missing indicators), standard scaling and
# MAGIC   one-hot encoding, all fitted on train inside the pipeline. No class weights, so the
# MAGIC   probabilities stay calibrated for the Brier score.
# MAGIC * LightGBM: missing values handled natively; 12-configuration grid
# MAGIC   (`num_leaves` × `min_child_samples` × `reg_lambda`), early stopping on validation AUC.
# MAGIC   Control and candidate get the same grid, so neither is favoured.
# MAGIC * The `*_noext` sensitivity models reuse the best parameters of their counterpart.
# MAGIC * `CODE_GENDER` is not used by any model (protected attribute); it is only a slice in
# MAGIC   the error analysis.

# COMMAND ----------

if W["experiment"]:
    EXPERIMENT = W["experiment"]
else:
    user = spark.sql("SELECT current_user()").first()[0]
    EXPERIMENT = f"/Users/{user}/hc_phase2_models"
mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()

record = modeling.RunRecord(
    feature_table=store.fq("gold", FEAT_TABLE), feature_table_delta_version=versions["features"],
    split_table=store.fq("gold", TVT_TABLE), split_table_delta_version=versions["split_tvt"],
    feature_version=features.FEATURE_VERSION, git_commit=GIT_COMMIT,
    seeds={"split_seed": features.SPLIT_SEED, "validation_seed": features.VALIDATION_SEED,
           "model_seed": modeling.MODEL_SEED, "bootstrap_seed": modeling.MODEL_SEED},
    libraries=modeling.library_versions(), split_counts=split_counts)
common_tags = {"git_commit": GIT_COMMIT, "feature_version": features.FEATURE_VERSION,
               "feature_table": record.feature_table,
               "feature_table_delta_version": str(versions["features"]),
               "split_table": record.split_table, "split_table_delta_version": str(versions["split_tvt"]),
               "feature_pipeline_git_commit": feat_props.get("hc.git_commit", "unknown"),
               "feature_pipeline_run_id": feat_props.get("hc.run_id", "unknown"),
               **{f"lib.{k}": v for k, v in record.libraries.items()}}

parent = mlflow.start_run(run_name=f"phase2_{time.strftime('%Y%m%d_%H%M')}")
mlflow.set_tags({**common_tags, "phase": "2", "role": "parent"})
mlflow.log_params({**record.seeds, "validation_fraction": features.VALIDATION_FRACTION,
                   "early_stopping_rounds": modeling.EARLY_STOPPING_ROUNDS,
                   "lgbm_grid": json.dumps(modeling.LGBM_GRID)})
mlflow.log_dict(checks, "input_checks.json")
record.mlflow_runs["parent"] = parent.info.run_id
print("experiment:", EXPERIMENT, "| parent run:", parent.info.run_id)

# COMMAND ----------

models, best_params, val_scores, tuning_rows = {}, {}, {}, []

def fit_and_log(spec, params, tag_extra=None, log_model=True):
    t0 = time.time()
    m = modeling.fit_model(spec, X["train"], y["train"], X["validation"], y["validation"], params)
    secs = time.time() - t0
    p_val = modeling.predict_proba(m, X["validation"])
    mv = modeling.metrics(y["validation"], p_val)
    with mlflow.start_run(run_name=spec.name, nested=True) as r:
        num, cat = spec.columns
        mlflow.set_tags({**common_tags, "model_name": spec.name, "role": spec.role,
                         "algo": spec.algo, "description": spec.description, **(tag_extra or {})})
        mlflow.log_params({**(params or {}), "algo": spec.algo, "groups": ",".join(spec.groups),
                           "include_ext": spec.include_ext, "n_features_in": len(num) + len(cat),
                           "best_iteration": modeling.best_iteration(m)})
        mlflow.log_metrics({f"val_{k}": v for k, v in mv.items()} | {"fit_seconds": secs})
        if log_model:
            mlflow.log_dict({"numeric": num, "categorical": cat}, "model_columns.json")
            sample = X["validation"].head(5)
            mlflow.sklearn.log_model(m, artifact_path="model",
                                     signature=infer_signature(sample, modeling.predict_proba(m, sample)),
                                     pyfunc_predict_fn="predict_proba")
    return m, mv, r.info.run_id

for spec in modeling.SPECS:
    if spec.role == "sensitivity":
        continue
    if spec.algo == "logreg":
        best_params[spec.name] = {}
    else:
        best = None
        for i, params in enumerate(modeling.lgbm_grid()):
            m, mv, rid = fit_and_log(spec, params, {"role": "tuning", "grid_index": str(i)},
                                     log_model=False)
            tuning_rows.append({"model": spec.name, **params, "best_iteration": modeling.best_iteration(m),
                                "val_roc_auc": mv["roc_auc"], "val_pr_auc": mv["pr_auc"],
                                "val_brier": mv["brier"], "run_id": rid})
            if best is None or mv["roc_auc"] > best[1]:
                best = (params, mv["roc_auc"])
            print(f"{spec.name} grid {i:2d} {params} -> val AUC {mv['roc_auc']:.4f} "
                  f"(iter {modeling.best_iteration(m)})")
        best_params[spec.name] = best[0]

counterpart = {"lgbm_app_noext": "lgbm_app", "lgbm_app_hist_noext": "lgbm_app_hist"}
for spec in modeling.SPECS:
    params = best_params.get(spec.name, best_params.get(counterpart.get(spec.name)))
    models[spec.name], val_scores[spec.name], record.mlflow_runs[spec.name] = fit_and_log(spec, params)

tuning = pd.DataFrame(tuning_rows)
val_table = pd.DataFrame(val_scores).T[["roc_auc", "pr_auc", "brier", "log_loss", "n"]]
display(val_table)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model choice (made on validation, before the test set is opened)
# MAGIC The deployed model is the best of baseline / control / candidate by **validation ROC-AUC**.
# MAGIC Sensitivity models are for the hypothesis write-up only.

# COMMAND ----------

eligible = [s.name for s in modeling.SPECS if s.role in ("baseline", "control", "candidate")]
CHOSEN = max(eligible, key=lambda n: val_scores[n]["roc_auc"])
mlflow.set_tags({"chosen_model": CHOSEN, "chosen_model_run_id": record.mlflow_runs[CHOSEN]})
print("chosen model:", CHOSEN, "| run", record.mlflow_runs[CHOSEN],
      "| params", best_params.get(CHOSEN, best_params.get(counterpart.get(CHOSEN))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Final evaluation on the test set
# MAGIC **From this cell on the test set is used. Models and choices above are frozen; if
# MAGIC anything above changes, the test results must be reported as a second look.**

# COMMAND ----------

pred_test = {n: modeling.predict_proba(m, X["test"]) for n, m in models.items()}
test_rows = []
for spec in modeling.SPECS:
    mt = modeling.metrics(y["test"], pred_test[spec.name])
    for k, v in mt.items():
        client.log_metric(record.mlflow_runs[spec.name], f"test_{k}", v)
    test_rows.append({"model": spec.name, "role": spec.role, "ext_sources": spec.include_ext,
                      **{f"val_{k}": val_scores[spec.name][k] for k in ("roc_auc", "pr_auc", "brier")},
                      **{f"test_{k}": mt[k] for k in ("roc_auc", "pr_auc", "brier", "log_loss")},
                      "mlflow_run_id": record.mlflow_runs[spec.name]})
results = pd.DataFrame(test_rows)
display(results)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hypothesis
# MAGIC *Adding borrowing-history features improves ROC-AUC by at least 0.03 compared with the
# MAGIC application form alone.* Main comparison: candidate vs control (same algorithm, same
# MAGIC split, same tuning). Rule fixed in advance (`modeling.hypothesis_verdict`):
# MAGIC **supported** if the test gain is ≥ 0.03 and its 95 % paired-bootstrap CI excludes 0.

# COMMAND ----------

comparisons = {
    "main: lgbm_app_hist vs lgbm_app": ("lgbm_app_hist", "lgbm_app"),
    "vs baseline: lgbm_app_hist vs logreg_app": ("lgbm_app_hist", "logreg_app"),
    "no EXT_SOURCE: lgbm_app_hist_noext vs lgbm_app_noext": ("lgbm_app_hist_noext", "lgbm_app_noext"),
}
hyp_rows = []
for label, (new, old) in comparisons.items():
    d = modeling.bootstrap_auc_delta(y["test"], pred_test[new], pred_test[old])
    hyp_rows.append({"comparison": label, **d, "verdict": modeling.hypothesis_verdict(d)})
hypothesis = pd.DataFrame(hyp_rows)
display(hypothesis)  # noqa: F821
main = hyp_rows[0]
mlflow.log_metrics({"hyp_delta_auc": main["delta"], "hyp_delta_ci_low": main["ci_low"],
                    "hyp_delta_ci_high": main["ci_high"]})
mlflow.set_tag("hypothesis_verdict", main["verdict"])
print(f"Hypothesis: {main['verdict']}  (ΔAUC = {main['delta']:+.4f}, "
      f"95% CI [{main['ci_low']:+.4f}, {main['ci_high']:+.4f}])")

# COMMAND ----------

# MAGIC %md ## 4. Error analysis (chosen model, test set)

# COMMAND ----------

ea = modeling.add_slices(parts["test"].assign(p=pred_test[CHOSEN]))
slices = modeling.slice_report(ea)
overall_auc = modeling.metrics(ea["TARGET"], ea["p"])["roc_auc"]
big = slices[slices["n"] >= 1000]
weakest = big.loc[big["roc_auc"].idxmin()]
print(f"overall test ROC-AUC {overall_auc:.4f}; weakest slice (n >= 1000): "
      f"{weakest['slice']} = {weakest['value']} -> ROC-AUC {weakest['roc_auc']:.4f}, "
      f"n = {weakest['n']}, default rate {weakest['default_rate']:.3f}, "
      f"mean prediction {weakest['mean_pred']:.3f}")
display(slices)  # noqa: F821

fn, fp = modeling.confident_errors(ea, k=5)
print("Confident false negatives (defaulted, lowest predicted risk):")
display(fn)  # noqa: F821
print("Confident false positives (repaid, highest predicted risk):")
display(fp)  # noqa: F821
importance = modeling.feature_importance(models[CHOSEN])
display(importance)  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 5. Figures for the README and slides

# COMMAND ----------

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for name in ("logreg_app", "lgbm_app", "lgbm_app_hist"):
    p = pred_test[name]
    fpr, tpr, _ = roc_curve(y["test"], p)
    auc = results.set_index("model").loc[name, "test_roc_auc"]
    ax[0].plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    prec, rec, _ = precision_recall_curve(y["test"], p)
    ax[1].plot(rec, prec, label=name)
    frac, mean_p = calibration_curve(y["test"], p, n_bins=10, strategy="quantile")
    ax[2].plot(mean_p, frac, marker="o", label=name)
ax[0].plot([0, 1], [0, 1], "k:", lw=1)
ax[2].plot([0, 0.5], [0, 0.5], "k:", lw=1)
ax[0].set(title="ROC (test)", xlabel="False positive rate", ylabel="True positive rate")
ax[1].set(title="Precision-recall (test)", xlabel="Recall", ylabel="Precision")
ax[2].set(title="Calibration (test, 10 quantile bins)", xlabel="Mean predicted probability",
          ylabel="Observed default rate")
for a in ax:
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{EVIDENCE}/test_curves.png", dpi=130)

fig2, ax2 = plt.subplots(figsize=(7, 6))
imp = importance.iloc[::-1]
ax2.barh(imp["feature"], imp["importance"])
ax2.set(title=f"{CHOSEN}: top features ({imp['kind'].iloc[0]})")
fig2.tight_layout()
fig2.savefig(f"{EVIDENCE}/feature_importance.png", dpi=130)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Export for registration and deployment (Lercia)
# MAGIC * `model_tags.json` — tags to put on the Azure ML model.
# MAGIC * `model_input_schema.json` — the endpoint's input contract (SK_ID_CURR + 50 features).
# MAGIC * `consistency_sample_100.json` — 100 fixed test rows (lowest `SK_ID_CURR`) and the
# MAGIC   probabilities of the **saved** model reloaded from MLflow, for the endpoint check.
# MAGIC * a fixed data snapshot (Parquet, Delta version in the name) and the MLflow model folder,
# MAGIC   copied to `export_dir` and logged to the parent run.

# COMMAND ----------

chosen_run = record.mlflow_runs[CHOSEN]
saved = mlflow.sklearn.load_model(f"runs:/{chosen_run}/model")
sample = parts["test"].nsmallest(100, "SK_ID_CURR").reset_index(drop=True)
Xs = modeling.to_model_frame(sample)
p_saved = modeling.predict_proba(saved, Xs)
p_memory = modeling.predict_proba(models[CHOSEN], Xs)
max_diff = float(np.max(np.abs(p_saved - p_memory)))
assert max_diff < 1e-12, max_diff

def _records(frame: pd.DataFrame) -> list[dict]:
    out = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in out]

consistency = {
    "model_name": CHOSEN, "training_run_id": chosen_run, "tolerance": 1e-6,
    "selection": "100 test rows with the lowest SK_ID_CURR",
    "inputs": _records(sample[["SK_ID_CURR", *features.FEATURE_NAMES]]),
    "expected": [{"SK_ID_CURR": int(i), "probability": float(p)}
                 for i, p in zip(sample["SK_ID_CURR"], p_saved)],
}
test_m = results.set_index("model").loc[CHOSEN]
tags = {
    "git_commit": GIT_COMMIT, "training_run_id": chosen_run, "mlflow_experiment": EXPERIMENT,
    "model_name": CHOSEN, "feature_version": features.FEATURE_VERSION,
    "feature_table": record.feature_table, "feature_table_delta_version": str(versions["features"]),
    "split_table": record.split_table, "split_table_delta_version": str(versions["split_tvt"]),
    "test_roc_auc": f"{test_m['test_roc_auc']:.4f}", "test_pr_auc": f"{test_m['test_pr_auc']:.4f}",
    "test_brier": f"{test_m['test_brier']:.4f}",
    "limitations": ("Trained on Kaggle Home Credit data (one lender, historical); EXT_SOURCE_* "
                    "timing unverified; CODE_GENDER excluded but proxies may remain; "
                    f"weakest slice {weakest['slice']}={weakest['value']} "
                    f"(AUC {weakest['roc_auc']:.3f}); not for automated decisions."),
}

files = {
    "model_tags.json": tags, "model_input_schema.json": modeling.input_schema(),
    "consistency_sample_100.json": consistency,
    "run_record.json": {**record.__dict__, "chosen_model": CHOSEN, "best_params": best_params,
                        "input_checks": checks, "hypothesis": hyp_rows},
}
for name, obj in files.items():
    with open(f"{EVIDENCE}/{name}", "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
results.to_csv(f"{EVIDENCE}/metrics_test.csv", index=False)
tuning.to_csv(f"{EVIDENCE}/tuning_validation.csv", index=False)
hypothesis.to_csv(f"{EVIDENCE}/hypothesis.csv", index=False)
slices.to_csv(f"{EVIDENCE}/slices_test.csv", index=False)
pd.concat([fn.assign(kind="false_negative"), fp.assign(kind="false_positive")]) \
    .to_csv(f"{EVIDENCE}/confident_errors.csv", index=False)
importance.to_csv(f"{EVIDENCE}/feature_importance.csv", index=False)
mlflow.log_artifacts(EVIDENCE, artifact_path="evidence")
print("wrote", sorted(os.listdir(EVIDENCE)))

# COMMAND ----------

# Fixed data snapshot (one versioned file, not a folder that can change) + model copy.
snap_name = (f"features_{features.FEATURE_VERSION}_delta{versions['features']}"
             f"_tvt{versions['split_tvt']}.parquet")
snap_local = f"/tmp/{snap_name}"
df.attrs = {}
df.to_parquet(snap_local, index=False)
snap_sha = hashlib.sha256(open(snap_local, "rb").read()).hexdigest()
mlflow.log_artifact(snap_local, artifact_path="data")
mlflow.set_tags({"data_snapshot": snap_name, "data_snapshot_sha256": snap_sha})

model_local = mlflow.artifacts.download_artifacts(f"runs:/{chosen_run}/model", dst_path="/tmp/hc_model")
if W["export_dir"]:
    import shutil
    dest = f"{W['export_dir']}/{chosen_run}"
    os.makedirs(dest, exist_ok=True)
    shutil.copy(snap_local, f"{dest}/{snap_name}")
    shutil.copytree(model_local, f"{dest}/model", dirs_exist_ok=True)
    for name in ("model_tags.json", "model_input_schema.json", "consistency_sample_100.json"):
        shutil.copy(f"{EVIDENCE}/{name}", f"{dest}/{name}")
    print("exported to", dest)
print("snapshot", snap_name, "sha256", snap_sha)

mlflow.end_run()
print(json.dumps({"parent_run": record.mlflow_runs["parent"], "chosen_model": CHOSEN,
                  "chosen_run": chosen_run, "verdict": main["verdict"]}, indent=2))

# COMMAND ----------


