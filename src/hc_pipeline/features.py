"""Feature set v1: one row per applicant (SK_ID_CURR).

Two feature groups, matching the hypothesis:
  * application - built only from the application form (baseline model inputs)
  * history     - aggregated from bureau, bureau_balance, previous_application,
                  installments_payments, POS_CASH_balance, credit_card_balance

`FEATURES` is the feature specification; docs/feature_spec.md is generated from it and
a unit test checks that the output columns match it exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

FEATURE_VERSION = "v1"
SPLIT_SEED = 3202
TEST_FRACTION = 0.20


@dataclass(frozen=True)
class Feature:
    name: str
    group: str          # application | history
    dtype: str          # double | int | string
    source: str
    definition: str
    leakage_note: str = "Known at application time."


_A, _H = "application", "history"
FEATURES: tuple[Feature, ...] = (
    # ---------------- application form (baseline group)
    Feature("AMT_INCOME_TOTAL", _A, "double", "application", "Declared annual income."),
    Feature("AMT_CREDIT", _A, "double", "application", "Credit amount requested."),
    Feature("AMT_ANNUITY", _A, "double", "application", "Loan annuity (periodic payment)."),
    Feature("AMT_GOODS_PRICE", _A, "double", "application", "Price of goods financed."),
    Feature("CREDIT_INCOME_RATIO", _A, "double", "application", "AMT_CREDIT / AMT_INCOME_TOTAL."),
    Feature("ANNUITY_INCOME_RATIO", _A, "double", "application",
            "AMT_ANNUITY / AMT_INCOME_TOTAL (payment burden)."),
    Feature("CREDIT_TERM", _A, "double", "application",
            "AMT_CREDIT / AMT_ANNUITY (approximate number of payments)."),
    Feature("GOODS_CREDIT_RATIO", _A, "double", "application",
            "AMT_GOODS_PRICE / AMT_CREDIT (share of credit covering goods)."),
    Feature("AGE_YEARS", _A, "double", "application", "-DAYS_BIRTH / 365.25."),
    Feature("EMPLOYED_YEARS", _A, "double", "application",
            "-DAYS_EMPLOYED / 365.25; NULL when DAYS_EMPLOYED was the 365243 placeholder."),
    Feature("DAYS_EMPLOYED_ANOM", _A, "int", "application",
            "1 if DAYS_EMPLOYED was the 365243 placeholder (mostly pensioners), else 0."),
    Feature("EXT_SOURCE_1", _A, "double", "application", "External score 1 (normalised).",
            "Third-party score; provenance unknown. Check it is computed before the application."),
    Feature("EXT_SOURCE_2", _A, "double", "application", "External score 2 (normalised).",
            "Third-party score; provenance unknown. Check it is computed before the application."),
    Feature("EXT_SOURCE_3", _A, "double", "application", "External score 3 (normalised).",
            "Third-party score; provenance unknown. Check it is computed before the application."),
    Feature("EXT_SOURCE_MEAN", _A, "double", "application",
            "Mean of the available EXT_SOURCE_1..3 (NULL if none).", "Same as EXT_SOURCE_*."),
    Feature("CNT_CHILDREN", _A, "double", "application", "Number of children."),
    Feature("CNT_FAM_MEMBERS", _A, "double", "application", "Family size."),
    Feature("REGION_RATING_CLIENT", _A, "double", "application", "Home Credit region rating (1-3)."),
    Feature("NAME_CONTRACT_TYPE", _A, "string", "application", "Cash or revolving loan."),
    Feature("CODE_GENDER", _A, "string", "application", "Gender ('XNA' -> NULL in silver).",
            "Protected attribute: kept for analysis; fairness review required before use in Phase 2."),
    Feature("NAME_INCOME_TYPE", _A, "string", "application", "Income type."),
    Feature("NAME_EDUCATION_TYPE", _A, "string", "application", "Highest education."),
    Feature("FLAG_OWN_CAR", _A, "string", "application", "Owns a car (Y/N)."),
    Feature("FLAG_OWN_REALTY", _A, "string", "application", "Owns real estate (Y/N)."),
    # ---------------- Credit Bureau
    Feature("BUR_LOAN_COUNT", _H, "int", "bureau", "Number of Credit Bureau credits (0 if none)."),
    Feature("BUR_ACTIVE_COUNT", _H, "int", "bureau", "Credits with CREDIT_ACTIVE = 'Active'."),
    Feature("BUR_ACTIVE_DEBT_SUM", _H, "double", "bureau", "Sum of AMT_CREDIT_SUM_DEBT over active credits."),
    Feature("BUR_DEBT_CREDIT_RATIO", _H, "double", "bureau",
            "Active debt / active credit amount (NULL without active credit)."),
    Feature("BUR_DEBT_INCOME_RATIO", _H, "double", "bureau + application",
            "BUR_ACTIVE_DEBT_SUM / AMT_INCOME_TOTAL."),
    Feature("BUR_OVERDUE_SUM", _H, "double", "bureau", "Sum of AMT_CREDIT_SUM_OVERDUE."),
    Feature("BUR_MAX_DAY_OVERDUE", _H, "double", "bureau", "Max CREDIT_DAY_OVERDUE."),
    Feature("BUR_DAYS_SINCE_LAST_CREDIT", _H, "double", "bureau",
            "-max(DAYS_CREDIT): days since the most recent bureau credit was opened."),
    Feature("BB_MONTHS_COUNT", _H, "int", "bureau_balance",
            "Months of bureau balance history with a DPD status (0-5)."),
    Feature("BB_LATE_MONTH_RATE", _H, "double", "bureau_balance",
            "Share of those months with STATUS 1-5 (any days past due)."),
    # ---------------- previous Home Credit applications
    Feature("PREV_APP_COUNT", _H, "int", "previous_application", "Number of previous applications."),
    Feature("PREV_REFUSED_COUNT", _H, "int", "previous_application",
            "Previous applications with NAME_CONTRACT_STATUS = 'Refused'."),
    Feature("PREV_APPROVED_RATE", _H, "double", "previous_application",
            "Approved / all previous applications."),
    Feature("PREV_CREDIT_APPLICATION_RATIO_MEAN", _H, "double", "previous_application",
            "Mean of AMT_CREDIT / AMT_APPLICATION (granted vs. asked)."),
    Feature("PREV_DAYS_SINCE_LAST_DECISION", _H, "double", "previous_application",
            "-max(DAYS_DECISION)."),
    # ---------------- repayment behaviour
    Feature("INST_COUNT", _H, "int", "installments_payments",
            "Number of past installments (payments grouped per installment)."),
    Feature("INST_LATE_RATE", _H, "double", "installments_payments",
            "Share of installments whose last payment came after the due day."),
    Feature("INST_DPD_MEAN", _H, "double", "installments_payments",
            "Mean days past due per installment (0 when on time)."),
    Feature("INST_UNDERPAID_RATE", _H, "double", "installments_payments",
            "Share of installments where total paid < amount due."),
    Feature("INST_PAID_RATIO", _H, "double", "installments_payments", "Total paid / total due."),
    Feature("POS_MONTHS_COUNT", _H, "int", "POS_CASH_balance", "Months of POS/cash loan history."),
    Feature("POS_LATE_MONTH_RATE", _H, "double", "POS_CASH_balance", "Share of months with SK_DPD > 0."),
    Feature("POS_DPD_MAX", _H, "double", "POS_CASH_balance", "Max SK_DPD."),
    Feature("CC_MONTHS_COUNT", _H, "int", "credit_card_balance", "Months of credit card history."),
    Feature("CC_UTILIZATION_MEAN", _H, "double", "credit_card_balance",
            "Mean of AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL (months with limit > 0)."),
    Feature("CC_DPD_MAX", _H, "double", "credit_card_balance", "Max SK_DPD on credit cards."),
)
FEATURE_NAMES = [f.name for f in FEATURES]
GROUPS = {g: [f.name for f in FEATURES if f.group == g] for g in (_A, _H)}
ID_COLUMNS = ["SK_ID_CURR", "TARGET", "DATASET"]


def _div(a, b):
    return F.when((b.isNotNull()) & (b != 0), a / b)


def _rate(cond):
    """Mean of a boolean, ignoring NULLs."""
    return F.avg(cond.cast("double"))


def application_features(train: DataFrame, test: DataFrame) -> DataFrame:
    app = (train.withColumn("DATASET", F.lit("train"))
           .unionByName(test.withColumn("DATASET", F.lit("test"))
                        .withColumn("TARGET", F.lit(None).cast("int")), allowMissingColumns=True))
    ext = [F.col(c) for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")]
    n_ext = sum(F.when(c.isNotNull(), 1).otherwise(0) for c in ext)
    s_ext = sum(F.coalesce(c, F.lit(0.0)) for c in ext)
    return app.select(
        "SK_ID_CURR", "TARGET", "DATASET",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        _div(F.col("AMT_CREDIT"), F.col("AMT_INCOME_TOTAL")).alias("CREDIT_INCOME_RATIO"),
        _div(F.col("AMT_ANNUITY"), F.col("AMT_INCOME_TOTAL")).alias("ANNUITY_INCOME_RATIO"),
        _div(F.col("AMT_CREDIT"), F.col("AMT_ANNUITY")).alias("CREDIT_TERM"),
        _div(F.col("AMT_GOODS_PRICE"), F.col("AMT_CREDIT")).alias("GOODS_CREDIT_RATIO"),
        (-F.col("DAYS_BIRTH") / 365.25).alias("AGE_YEARS"),
        (-F.col("DAYS_EMPLOYED") / 365.25).alias("EMPLOYED_YEARS"),
        F.col("DAYS_EMPLOYED_ANOM").cast("int").alias("DAYS_EMPLOYED_ANOM"),
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        F.when(n_ext > 0, s_ext / n_ext).alias("EXT_SOURCE_MEAN"),
        "CNT_CHILDREN", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT",
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    )


def history_features(t: dict[str, DataFrame]) -> dict[str, DataFrame]:
    """Per-applicant aggregates. Only records dated on/before the application day
    (relative day/month <= 0) are used -- see docs/feature_spec.md, leakage section."""
    bureau = t["bureau"].filter("DAYS_CREDIT <= 0")
    active = F.col("CREDIT_ACTIVE") == "Active"
    bur = bureau.groupBy("SK_ID_CURR").agg(
        F.count("*").alias("BUR_LOAN_COUNT"),
        F.sum(active.cast("int")).alias("BUR_ACTIVE_COUNT"),
        F.sum(F.when(active, F.col("AMT_CREDIT_SUM_DEBT"))).alias("BUR_ACTIVE_DEBT_SUM"),
        F.sum(F.when(active, F.col("AMT_CREDIT_SUM"))).alias("_active_credit"),
        F.sum("AMT_CREDIT_SUM_OVERDUE").alias("BUR_OVERDUE_SUM"),
        F.max("CREDIT_DAY_OVERDUE").cast("double").alias("BUR_MAX_DAY_OVERDUE"),
        (-F.max("DAYS_CREDIT")).cast("double").alias("BUR_DAYS_SINCE_LAST_CREDIT"),
    )
    bur = (bur.withColumn("BUR_DEBT_CREDIT_RATIO", _div(F.col("BUR_ACTIVE_DEBT_SUM"), F.col("_active_credit")))
           .drop("_active_credit"))

    bb = (t["bureau_balance"].filter("MONTHS_BALANCE <= 0")
          .filter(F.col("STATUS").isin("0", "1", "2", "3", "4", "5"))
          .join(bureau.select("SK_ID_BUREAU", "SK_ID_CURR"), "SK_ID_BUREAU")
          .groupBy("SK_ID_CURR").agg(
              F.count("*").alias("BB_MONTHS_COUNT"),
              _rate(F.col("STATUS") != "0").alias("BB_LATE_MONTH_RATE")))

    prev = t["previous_application"].filter("DAYS_DECISION <= 0")
    status = F.col("NAME_CONTRACT_STATUS")
    pa = prev.groupBy("SK_ID_CURR").agg(
        F.count("*").alias("PREV_APP_COUNT"),
        F.sum((status == "Refused").cast("int")).alias("PREV_REFUSED_COUNT"),
        _rate(status == "Approved").alias("PREV_APPROVED_RATE"),
        F.avg(_div(F.col("AMT_CREDIT"), F.col("AMT_APPLICATION"))).alias("PREV_CREDIT_APPLICATION_RATIO_MEAN"),
        (-F.max("DAYS_DECISION")).cast("double").alias("PREV_DAYS_SINCE_LAST_DECISION"),
    )

    # Payments can be split over several rows: aggregate to one row per installment first.
    ip = t["installments_payments"].filter("DAYS_INSTALMENT <= 0")
    per_inst = (ip.withColumn("_pay_day", F.when(F.col("DAYS_ENTRY_PAYMENT") <= 0, F.col("DAYS_ENTRY_PAYMENT")))
                .groupBy("SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER")
                .agg(F.max("AMT_INSTALMENT").alias("due"),
                     F.sum("AMT_PAYMENT").alias("paid"),
                     F.min("DAYS_INSTALMENT").alias("due_day"),
                     F.max("_pay_day").alias("pay_day")))
    dpd = F.greatest(F.col("pay_day") - F.col("due_day"), F.lit(0))
    inst = per_inst.groupBy("SK_ID_CURR").agg(
        F.count("*").alias("INST_COUNT"),
        _rate(F.col("pay_day") > F.col("due_day")).alias("INST_LATE_RATE"),
        F.avg(dpd).cast("double").alias("INST_DPD_MEAN"),
        _rate(F.coalesce(F.col("paid"), F.lit(0.0)) < F.col("due") - 0.01).alias("INST_UNDERPAID_RATE"),
        _div(F.sum("paid"), F.sum("due")).alias("INST_PAID_RATIO"),
    )

    pos = t["pos_cash_balance"].filter("MONTHS_BALANCE <= 0").groupBy("SK_ID_CURR").agg(
        F.count("*").alias("POS_MONTHS_COUNT"),
        _rate(F.col("SK_DPD") > 0).alias("POS_LATE_MONTH_RATE"),
        F.max("SK_DPD").cast("double").alias("POS_DPD_MAX"))

    cc_src = t["credit_card_balance"].filter("MONTHS_BALANCE <= 0")
    cc = cc_src.groupBy("SK_ID_CURR").agg(
        F.count("*").alias("CC_MONTHS_COUNT"),
        F.avg(F.when(F.col("AMT_CREDIT_LIMIT_ACTUAL") > 0,
                     F.col("AMT_BALANCE") / F.col("AMT_CREDIT_LIMIT_ACTUAL"))).alias("CC_UTILIZATION_MEAN"),
        F.max("SK_DPD").cast("double").alias("CC_DPD_MAX"))
    return {"bureau": bur, "bureau_balance": bb, "previous_application": pa,
            "installments_payments": inst, "pos_cash_balance": pos, "credit_card_balance": cc}


def build_features(t: dict[str, DataFrame]) -> DataFrame:
    """t: silver DataFrames keyed by table name. Returns one row per SK_ID_CURR."""
    app = application_features(t["application_train"], t["application_test"])
    out = app
    for agg in history_features(t).values():
        out = out.join(agg, "SK_ID_CURR", "left")
    # No history rows means a true count of 0; rates/ratios stay NULL (undefined).
    counts = [f.name for f in FEATURES if f.name.endswith("_COUNT")]
    out = out.fillna(0, subset=counts)
    out = out.withColumn("BUR_DEBT_INCOME_RATIO",
                         _div(F.col("BUR_ACTIVE_DEBT_SUM"), F.col("AMT_INCOME_TOTAL")))
    casts = {f.name: f.dtype for f in FEATURES}
    return out.select(*ID_COLUMNS, *[F.col(n).cast(casts[n]).alias(n) for n in FEATURE_NAMES])


def stratified_split(features: DataFrame, seed: int = SPLIT_SEED,
                     test_fraction: float = TEST_FRACTION) -> DataFrame:
    """Deterministic stratified 80/20 split of the labelled rows.

    Rows are ordered by a seeded hash of SK_ID_CURR inside each TARGET class, and the
    first `test_fraction` of each class goes to test. Same input -> same split, on any
    cluster size, so the baseline and comparison models share it exactly.
    """
    lab = features.filter(F.col("DATASET") == "train").select("SK_ID_CURR", "TARGET")
    w = Window.partitionBy("TARGET").orderBy(F.xxhash64(F.col("SK_ID_CURR"), F.lit(seed)), "SK_ID_CURR")
    return (lab.withColumn("_r", F.percent_rank().over(w))
            .withColumn("SPLIT", F.when(F.col("_r") < test_fraction, "test").otherwise("train"))
            .drop("_r"))


# Phase 2: validation carved out of the Phase-1 train split (test is never touched).
VALIDATION_SEED = 3203
VALIDATION_FRACTION = 0.20


def validation_split(split: DataFrame, seed: int = VALIDATION_SEED,
                     fraction: float = VALIDATION_FRACTION) -> DataFrame:
    """Train / validation / test from the Phase-1 80/20 split.

    Same rule as `stratified_split` (seeded hash of SK_ID_CURR, ordered inside each TARGET
    class) with a different seed, applied only to SPLIT = 'train'. Rows with
    SPLIT = 'test' are passed through unchanged, so the Phase-1 test set stays identical.
    """
    tr = split.filter(F.col("SPLIT") == "train")
    w = Window.partitionBy("TARGET").orderBy(F.xxhash64(F.col("SK_ID_CURR"), F.lit(seed)), "SK_ID_CURR")
    tr = (tr.withColumn("_r", F.percent_rank().over(w))
          .withColumn("SPLIT", F.when(F.col("_r") < fraction, "validation").otherwise("train"))
          .drop("_r"))
    return tr.unionByName(split.filter(F.col("SPLIT") == "test"))
