"""Phase 2: model pipelines, tuning, metrics and error analysis (pandas / scikit-learn).

Everything here works on pandas DataFrames with the columns of `gold_features_v1`, so the
same code runs in the Databricks notebook and in the unit tests.

The saved model is a plain scikit-learn `Pipeline` (preprocessing + classifier) built only
from scikit-learn and LightGBM classes. The endpoint therefore needs only those libraries,
not this package, and every fitted transform (imputer, scaler, encoder) is fitted on the
training split only, inside the pipeline.
"""
from __future__ import annotations

import itertools
import platform
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hc_pipeline import features

MODEL_SEED = 3202

# CODE_GENDER is a protected attribute (docs/feature_spec.md). It stays in the input
# contract (the endpoint accepts all 50 fields) but no model uses it; it is only used to
# check model quality per group in the error analysis.
EXCLUDED_FROM_MODEL = ("CODE_GENDER",)
EXT_SOURCES = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_MEAN")

_DTYPE = {f.name: f.dtype for f in features.FEATURES}
CATEGORICAL = [f.name for f in features.FEATURES if f.dtype == "string"]


def model_columns(groups: tuple[str, ...] = ("application", "history"),
                  include_ext: bool = True) -> tuple[list[str], list[str]]:
    """(numeric, categorical) columns a model uses, in feature-spec order."""
    cols = [c for g in groups for c in features.GROUPS[g]
            if c not in EXCLUDED_FROM_MODEL and (include_ext or c not in EXT_SOURCES)]
    return [c for c in cols if c not in CATEGORICAL], [c for c in cols if c in CATEGORICAL]


@dataclass(frozen=True)
class ModelSpec:
    """One model in the experiment. `role` is baseline | control | candidate | sensitivity."""
    name: str
    algo: str                       # logreg | lightgbm
    groups: tuple[str, ...]
    role: str
    include_ext: bool = True
    description: str = ""

    @property
    def columns(self) -> tuple[list[str], list[str]]:
        return model_columns(self.groups, self.include_ext)


SPECS = (
    ModelSpec("logreg_app", "logreg", ("application",), "baseline",
              description="Logistic regression, application form only (Phase-1 baseline plan)."),
    ModelSpec("lgbm_app", "lightgbm", ("application",), "control",
              description="LightGBM, application form only. Same algorithm as the candidate, "
                          "so the difference to it measures the history features."),
    ModelSpec("lgbm_app_hist", "lightgbm", ("application", "history"), "candidate",
              description="LightGBM, application form + borrowing history."),
    ModelSpec("lgbm_app_noext", "lightgbm", ("application",), "sensitivity", include_ext=False,
              description="Control without EXT_SOURCE_* (their timing is unverified)."),
    ModelSpec("lgbm_app_hist_noext", "lightgbm", ("application", "history"), "sensitivity",
              include_ext=False,
              description="Candidate without EXT_SOURCE_* (their timing is unverified)."),
)
SPEC_BY_NAME = {s.name: s for s in SPECS}


# --------------------------------------------------------------------------- inputs
def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All 50 features with stable pandas dtypes: float64 for numbers (NULL -> NaN),
    object for strings (NULL -> None). The endpoint must build its input the same way."""
    out = pd.DataFrame(index=df.index)
    for c in features.FEATURE_NAMES:
        if _DTYPE[c] == "string":
            s = df[c].astype(object)
            out[c] = s.where(s.notna(), None)
        else:
            out[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
    return out


def input_schema() -> list[dict]:
    """The endpoint's input contract: SK_ID_CURR plus the 50 features, with JSON types."""
    json_type = {"double": "number", "int": "integer", "string": "string"}
    return ([{"name": "SK_ID_CURR", "type": "integer", "nullable": False}]
            + [{"name": f.name, "type": json_type[f.dtype], "nullable": True,
                "used_by_model": f.name not in EXCLUDED_FROM_MODEL, "group": f.group}
               for f in features.FEATURES])


# --------------------------------------------------------------------------- pipelines
def _preprocessor(algo: str, num: list[str], cat: list[str]) -> ColumnTransformer:
    # OneHotEncoder treats None as its own category and ignores unseen categories at
    # prediction time, so new or missing categories never break scoring.
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if algo == "logreg":
        numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scale", StandardScaler())])
    else:
        numeric = "passthrough"     # LightGBM handles NaN natively
    return ColumnTransformer([("num", numeric, num), ("cat", onehot, cat)],
                             remainder="drop", verbose_feature_names_out=False)


LOGREG_PARAMS = {"C": 1.0, "max_iter": 2000}
LGBM_FIXED = {"objective": "binary", "n_estimators": 3000, "learning_rate": 0.03,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
              "deterministic": True, "force_row_wise": True, "verbose": -1}
# Small grid, judged on the validation set only (12 configurations).
LGBM_GRID = {"num_leaves": [15, 31, 63], "min_child_samples": [50, 200], "reg_lambda": [0.0, 5.0]}
EARLY_STOPPING_ROUNDS = 100


def lgbm_grid() -> list[dict]:
    keys = list(LGBM_GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*LGBM_GRID.values())]


def fit_model(spec: ModelSpec, X_train: pd.DataFrame, y_train, X_val: pd.DataFrame | None = None,
              y_val=None, params: dict | None = None, seed: int = MODEL_SEED) -> Pipeline:
    """Fit preprocessing + classifier on the training split only.

    LightGBM uses the validation split for early stopping (model selection only; nothing
    is fitted on it). The returned pipeline predicts with the best iteration.
    """
    num, cat = spec.columns
    pre = _preprocessor(spec.algo, num, cat).set_output(transform="pandas")
    Xt = pre.fit_transform(X_train)
    if spec.algo == "logreg":
        clf = LogisticRegression(**{**LOGREG_PARAMS, **(params or {})}, random_state=seed)
        clf.fit(Xt, y_train)
    else:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(**{**LGBM_FIXED, **(params or {})}, random_state=seed)
        if X_val is None:
            clf.fit(Xt, y_train)
        else:
            clf.fit(Xt, y_train, eval_set=[(pre.transform(X_val), y_val)], eval_metric="auc",
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    return Pipeline([("prep", pre), ("model", clf)])


def predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def best_iteration(model: Pipeline) -> int | None:
    return getattr(model.named_steps["model"], "best_iteration_", None)


# --------------------------------------------------------------------------- metrics
def metrics(y, p) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))),
            "base_rate": float(y.mean()), "mean_pred": float(p.mean()), "n": int(len(y))}


def bootstrap_auc_delta(y, p_new, p_old, n_boot: int = 1000, seed: int = MODEL_SEED) -> dict:
    """Paired bootstrap of ROC-AUC(p_new) - ROC-AUC(p_old) on the same rows."""
    y, p_new, p_old = map(np.asarray, (y, p_new, p_old))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        deltas.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_old[i]))
    d = np.array(deltas)
    return {"delta": float(roc_auc_score(y, p_new) - roc_auc_score(y, p_old)),
            "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "n_boot": int(len(d))}


HYPOTHESIS_MIN_GAIN = 0.03


def hypothesis_verdict(delta: dict, min_gain: float = HYPOTHESIS_MIN_GAIN) -> str:
    """Decision rule, fixed before looking at the test set:
    supported  - point gain >= min_gain and the 95 % CI excludes 0
    refuted    - point gain < min_gain (the CI is reported either way)."""
    if delta["delta"] >= min_gain and delta["ci_low"] > 0:
        return "supported"
    if delta["delta"] >= min_gain:
        return "inconclusive (gain >= threshold but CI includes 0)"
    return "refuted"


# --------------------------------------------------------------------------- error analysis
def add_slices(df: pd.DataFrame) -> pd.DataFrame:
    """Slice columns for the error analysis (computed from features, not used by models)."""
    out = df.copy()
    out["slice_age"] = pd.cut(out["AGE_YEARS"], [0, 30, 40, 50, 60, 200], right=False,
                              labels=["<30", "30-39", "40-49", "50-59", "60+"]).astype(str)
    no_hist = (out["BUR_LOAN_COUNT"].fillna(0) == 0) & (out["PREV_APP_COUNT"].fillna(0) == 0)
    out["slice_history"] = np.where(no_hist, "no bureau or previous loans", "has history")
    n_ext = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].notna().sum(axis=1)
    out["slice_ext_available"] = n_ext.astype(str) + " of 3"
    return out


SLICE_COLUMNS = ("NAME_CONTRACT_TYPE", "slice_age", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                 "CODE_GENDER", "slice_history", "slice_ext_available")


def slice_report(df: pd.DataFrame, y_col: str = "TARGET", p_col: str = "p",
                 slices=SLICE_COLUMNS, min_n: int = 500) -> pd.DataFrame:
    """ROC-AUC, PR-AUC, Brier and calibration gap for every value of every slice column."""
    rows = []
    for col in slices:
        for val, g in df.groupby(df[col].astype(str).fillna("NULL"), dropna=False):
            if len(g) < min_n or g[y_col].nunique() < 2:
                continue
            m = metrics(g[y_col], g[p_col])
            rows.append({"slice": col, "value": val, "n": m["n"], "default_rate": m["base_rate"],
                         "mean_pred": m["mean_pred"], "roc_auc": m["roc_auc"],
                         "pr_auc": m["pr_auc"], "brier": m["brier"]})
    return pd.DataFrame(rows).sort_values(["slice", "roc_auc"]).reset_index(drop=True)


ERROR_COLUMNS = ("SK_ID_CURR", "TARGET", "p", "EXT_SOURCE_MEAN", "AGE_YEARS", "NAME_INCOME_TYPE",
                 "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "BUR_LOAN_COUNT",
                 "INST_LATE_RATE", "PREV_REFUSED_COUNT")


def confident_errors(df: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(false negatives, false positives): defaulters with the lowest predicted risk and
    non-defaulters with the highest."""
    cols = [c for c in ERROR_COLUMNS if c in df.columns]
    fn = df[df["TARGET"] == 1].nsmallest(k, "p")[cols]
    fp = df[df["TARGET"] == 0].nlargest(k, "p")[cols]
    return fn.reset_index(drop=True), fp.reset_index(drop=True)


def feature_importance(model: Pipeline, top: int = 20) -> pd.DataFrame:
    clf = model.named_steps["model"]
    names = model.named_steps["prep"].get_feature_names_out()
    if hasattr(clf, "booster_"):
        imp = clf.booster_.feature_importance(importance_type="gain")
        kind = "gain"
    else:
        imp, kind = np.abs(clf.coef_[0]), "abs_coef (standardised)"
    return (pd.DataFrame({"feature": names, "importance": imp, "kind": kind})
            .sort_values("importance", ascending=False).head(top).reset_index(drop=True))


# --------------------------------------------------------------------------- traceability
def library_versions() -> dict[str, str]:
    import sklearn, lightgbm, mlflow  # noqa: E401
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
            "mlflow": mlflow.__version__}


@dataclass
class RunRecord:
    """Everything needed to reproduce a training run; logged to MLflow and written to
    docs/evidence/phase2/run_record.json."""
    feature_table: str
    feature_table_delta_version: str | None
    split_table: str
    split_table_delta_version: str | None
    feature_version: str
    git_commit: str
    seeds: dict
    libraries: dict
    split_counts: dict
    mlflow_runs: dict = field(default_factory=dict)
