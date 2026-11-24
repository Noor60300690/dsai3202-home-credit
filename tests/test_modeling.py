"""Phase 2 tests: validation split, model inputs, pipelines and the hypothesis rule."""
import numpy as np
import pandas as pd
import pytest
from pyspark.sql import functions as F

from hc_pipeline import features, modeling


# --------------------------------------------------------------------------- split
def _split(spark, n=2000):
    rows = [(i, int(i % 10 == 0)) for i in range(1, n + 1)]
    f = spark.createDataFrame([(i, t, "train") for i, t in rows],
                              "SK_ID_CURR bigint, TARGET int, DATASET string")
    return features.stratified_split(f)


def test_validation_split_keeps_test_and_is_stratified(spark):
    base = _split(spark)
    tvt = features.validation_split(base)
    assert tvt.count() == base.count()
    test_before = {r[0] for r in base.filter("SPLIT = 'test'").select("SK_ID_CURR").collect()}
    test_after = {r[0] for r in tvt.filter("SPLIT = 'test'").select("SK_ID_CURR").collect()}
    assert test_before == test_after                      # Phase-1 test set untouched
    c = {(r["SPLIT"], r["TARGET"]): r["count"] for r in tvt.groupBy("SPLIT", "TARGET").count().collect()}
    train_n = {t: c[("train", t)] + c[("validation", t)] for t in (0, 1)}
    for t in (0, 1):                                      # 20 % of each class -> validation
        assert abs(c[("validation", t)] / train_n[t] - 0.20) < 0.01


def test_validation_split_is_deterministic_and_seed_dependent(spark):
    base = _split(spark)
    ids = lambda df: sorted(r[0] for r in df.filter("SPLIT = 'validation'")  # noqa: E731
                            .select("SK_ID_CURR").collect())
    assert ids(features.validation_split(base)) == ids(features.validation_split(base))
    assert ids(features.validation_split(base)) != ids(features.validation_split(base, seed=1))
    assert base.filter(F.col("SPLIT") == "validation").count() == 0


# --------------------------------------------------------------------------- models
def _frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    d = {}
    for f in features.FEATURES:
        if f.dtype == "string":
            d[f.name] = rng.choice(["A", "B", None], n)
        else:
            v = rng.normal(size=n)
            v[rng.random(n) < 0.2] = np.nan
            d[f.name] = v
    df = pd.DataFrame(d)
    signal = np.nan_to_num(df["INST_LATE_RATE"]) - np.nan_to_num(df["EXT_SOURCE_2"])
    df["TARGET"] = (signal + rng.normal(scale=0.8, size=n) > 1.3).astype(int)
    df["SK_ID_CURR"] = np.arange(n)
    return df


def test_model_columns_exclude_gender_and_ext():
    num, cat = modeling.model_columns(("application", "history"))
    assert "CODE_GENDER" not in num + cat and len(num + cat) == 49
    num, cat = modeling.model_columns(("application",), include_ext=False)
    assert not set(modeling.EXT_SOURCES) & set(num + cat)
    assert not set(features.GROUPS["history"]) & set(num + cat)


@pytest.mark.parametrize("name", ["logreg_app", "lgbm_app_hist"])
def test_pipeline_fits_scores_and_handles_missing_and_unseen(name):
    df = _frame()
    X, y = modeling.to_model_frame(df), df["TARGET"]
    spec = modeling.SPEC_BY_NAME[name]
    m = modeling.fit_model(spec, X[:2000], y[:2000], X[2000:2500], y[2000:2500],
                           params={"num_leaves": 7} if spec.algo == "lightgbm" else None)
    p = modeling.predict_proba(m, X[2500:])
    assert p.shape == (500,) and np.all((p >= 0) & (p <= 1))
    assert modeling.metrics(y[2500:], p)["roc_auc"] > 0.6
    odd = X[2500:2501].copy()                       # all missing + unseen category
    odd.loc[:, :] = np.nan
    odd["NAME_CONTRACT_TYPE"] = "never seen"
    odd = modeling.to_model_frame(odd)
    assert np.isfinite(modeling.predict_proba(m, odd)).all()


def test_model_does_not_use_gender():
    df = _frame()
    X, y = modeling.to_model_frame(df), df["TARGET"]
    m = modeling.fit_model(modeling.SPEC_BY_NAME["lgbm_app_hist"], X[:2000], y[:2000],
                           X[2000:2500], y[2000:2500], params={"num_leaves": 7})
    flipped = X[2500:].copy()
    flipped["CODE_GENDER"] = np.where(flipped["CODE_GENDER"] == "A", "B", "A")
    assert np.allclose(modeling.predict_proba(m, X[2500:]), modeling.predict_proba(m, flipped))


def test_hypothesis_rule():
    assert modeling.hypothesis_verdict({"delta": 0.04, "ci_low": 0.02}) == "supported"
    assert modeling.hypothesis_verdict({"delta": 0.02, "ci_low": 0.01}) == "refuted"
    assert modeling.hypothesis_verdict({"delta": 0.035, "ci_low": -0.01}).startswith("inconclusive")


def test_input_schema_lists_id_and_50_features():
    s = modeling.input_schema()
    assert s[0]["name"] == "SK_ID_CURR" and len(s) == 51
    assert [c["name"] for c in s[1:]] == features.FEATURE_NAMES
