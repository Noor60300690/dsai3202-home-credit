"""Feature v1 tests: output matches the spec, leakage guard, split is stratified and stable."""
from pyspark.sql import functions as F

from hc_pipeline import features
from hc_pipeline.features import FEATURE_NAMES, ID_COLUMNS


def _tables(spark):
    app = ("SK_ID_CURR bigint, TARGET int, AMT_INCOME_TOTAL double, AMT_CREDIT double, "
           "AMT_ANNUITY double, AMT_GOODS_PRICE double, DAYS_BIRTH int, DAYS_EMPLOYED int, "
           "DAYS_EMPLOYED_ANOM int, EXT_SOURCE_1 double, EXT_SOURCE_2 double, EXT_SOURCE_3 double, "
           "CNT_CHILDREN double, CNT_FAM_MEMBERS double, REGION_RATING_CLIENT double, "
           "NAME_CONTRACT_TYPE string, CODE_GENDER string, NAME_INCOME_TYPE string, "
           "NAME_EDUCATION_TYPE string, FLAG_OWN_CAR string, FLAG_OWN_REALTY string")
    row = lambda i, t: (i, t, 100000.0, 200000.0, 10000.0, 180000.0, -10957, None, 1,  # noqa: E731
                        None, 0.4, 0.6, 0.0, 1.0, 2.0, "Cash loans", "F", "Working", "Higher", "N", "Y")
    train = spark.createDataFrame([row(1, 0), row(2, 1)], app)
    test = spark.createDataFrame([row(3, None)], app).drop("TARGET")
    bureau = spark.createDataFrame(
        [(1, 10, "Active", -100, 0.0, 1000.0, 400.0, 0.0),
         (1, 11, "Closed", -900, 0.0, 500.0, 0.0, 0.0),
         (1, 12, "Active", 5, 0.0, 9e9, 9e9, 0.0)],          # dated AFTER application -> ignored
        "SK_ID_CURR bigint, SK_ID_BUREAU bigint, CREDIT_ACTIVE string, DAYS_CREDIT int, "
        "CREDIT_DAY_OVERDUE double, AMT_CREDIT_SUM double, AMT_CREDIT_SUM_DEBT double, "
        "AMT_CREDIT_SUM_OVERDUE double")
    bb = spark.createDataFrame([(10, -1, "0"), (10, -2, "1"), (10, -3, "C"), (11, -1, "X")],
                               "SK_ID_BUREAU bigint, MONTHS_BALANCE int, STATUS string")
    prev = spark.createDataFrame([(100, 1, "Approved", 1000.0, 900.0, -50),
                                  (101, 1, "Refused", 1000.0, 0.0, -20)],
                                 "SK_ID_PREV bigint, SK_ID_CURR bigint, NAME_CONTRACT_STATUS string, "
                                 "AMT_APPLICATION double, AMT_CREDIT double, DAYS_DECISION int")
    # installment 1 paid in two parts, on time; installment 2 paid 10 days late
    inst = spark.createDataFrame([(100, 1, 1.0, 1.0, -40, -45, 100.0, 60.0),
                                  (100, 1, 1.0, 1.0, -40, -41, 100.0, 40.0),
                                  (100, 1, 1.0, 2.0, -10, 0, 100.0, 100.0)],
                                 "SK_ID_PREV bigint, SK_ID_CURR bigint, NUM_INSTALMENT_VERSION double, "
                                 "NUM_INSTALMENT_NUMBER double, DAYS_INSTALMENT int, DAYS_ENTRY_PAYMENT int, "
                                 "AMT_INSTALMENT double, AMT_PAYMENT double")
    pos = spark.createDataFrame([(100, 1, -1, 0.0), (100, 1, -2, 3.0)],
                                "SK_ID_PREV bigint, SK_ID_CURR bigint, MONTHS_BALANCE int, SK_DPD double")
    cc = spark.createDataFrame([(200, 2, -1, 50.0, 100.0, 0.0), (200, 2, -2, 0.0, 0.0, 0.0)],
                               "SK_ID_PREV bigint, SK_ID_CURR bigint, MONTHS_BALANCE int, AMT_BALANCE double, "
                               "AMT_CREDIT_LIMIT_ACTUAL double, SK_DPD double")
    return {"application_train": train, "application_test": test, "bureau": bureau,
            "bureau_balance": bb, "previous_application": prev, "installments_payments": inst,
            "pos_cash_balance": pos, "credit_card_balance": cc}


def test_output_matches_spec(spark):
    f = features.build_features(_tables(spark))
    assert f.columns == ID_COLUMNS + FEATURE_NAMES
    assert f.count() == 3 and f.select("SK_ID_CURR").distinct().count() == 3
    spec_types = {x.name: x.dtype for x in features.FEATURES}
    assert all(t == spec_types[c] for c, t in f.dtypes if c in spec_types)


def test_feature_values_and_leakage_guard(spark):
    r = {x["SK_ID_CURR"]: x for x in features.build_features(_tables(spark)).collect()}
    a = r[1]
    assert a["CREDIT_INCOME_RATIO"] == 2.0 and a["AGE_YEARS"] > 29.9
    assert a["EMPLOYED_YEARS"] is None and a["DAYS_EMPLOYED_ANOM"] == 1
    assert a["EXT_SOURCE_MEAN"] == 0.5                         # mean of available scores
    assert a["BUR_LOAN_COUNT"] == 2                            # future-dated credit excluded
    assert a["BUR_ACTIVE_DEBT_SUM"] == 400.0 and a["BUR_DEBT_CREDIT_RATIO"] == 0.4
    assert a["BB_MONTHS_COUNT"] == 2 and a["BB_LATE_MONTH_RATE"] == 0.5   # C/X ignored
    assert a["PREV_APP_COUNT"] == 2 and a["PREV_REFUSED_COUNT"] == 1
    assert a["INST_COUNT"] == 2                                # split payment counted once
    assert a["INST_LATE_RATE"] == 0.5 and a["INST_UNDERPAID_RATE"] == 0.0
    assert a["POS_LATE_MONTH_RATE"] == 0.5 and a["CC_MONTHS_COUNT"] == 0
    b = r[2]                                                   # no history at all
    assert b["BUR_LOAN_COUNT"] == 0 and b["BUR_DEBT_CREDIT_RATIO"] is None
    assert b["CC_UTILIZATION_MEAN"] == 0.5                     # zero-limit month skipped
    assert r[3]["DATASET"] == "test" and r[3]["TARGET"] is None


def test_split_is_stratified_and_deterministic(spark):
    n = 2000
    df = spark.range(n).select(F.col("id").alias("SK_ID_CURR"),
                               (F.col("id") % 12 == 0).cast("int").alias("TARGET"),
                               F.lit("train").alias("DATASET"))
    s1 = features.stratified_split(df)
    s2 = features.stratified_split(df.repartition(7))
    assert s1.count() == n
    rates = {r["SPLIT"]: r["rate"] for r in s1.groupBy("SPLIT").agg(F.avg("TARGET").alias("rate")).collect()}
    assert abs(rates["train"] - rates["test"]) < 0.01
    frac = s1.filter("SPLIT = 'test'").count() / n
    assert abs(frac - 0.2) < 0.01
    assert s1.join(s2, ["SK_ID_CURR", "SPLIT"]).count() == n   # same assignment
