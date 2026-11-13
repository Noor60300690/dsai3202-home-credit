"""Schema contract for the Home Credit Default Risk tables.

Single source of truth used by the ETL, the quality gate, the data dictionary
and the Unity Catalog registration. Column types are resolved by explicit
override first, then by Home Credit naming conventions (see `resolve_type`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Placeholder Home Credit uses for "no date" in DAYS_* columns (~1000 years).
DAYS_PLACEHOLDER = 365243

# Columns that are categorical even though the name pattern would suggest numeric.
STRING_COLUMNS = {
    "ORGANIZATION_TYPE", "OCCUPATION_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
    "CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE", "STATUS",
    "FLAG_LAST_APPL_PER_CONTRACT", "CHANNEL_TYPE", "PRODUCT_COMBINATION",
}
STRING_PREFIXES = ("NAME_", "CODE_", "WEEKDAY_")
INT_PREFIXES = ("DAYS_", "MONTHS_BALANCE")
# Day counts that are not always whole numbers in the real data (found by
# scripts/build_data_dictionary.py): application_train SK_ID_CURR 408583 has
# DAYS_REGISTRATION = -10116.0417 (10116 days + 1 hour). Typed as DOUBLE so the
# applicant is kept instead of quarantined.
DOUBLE_COLUMNS = {"DAYS_REGISTRATION"}


def resolve_type(column: str) -> str:
    """Return the silver Spark SQL type for a raw column name."""
    if column in STRING_COLUMNS or column.startswith(STRING_PREFIXES):
        return "string"
    if column.startswith("SK_ID_"):
        return "bigint"
    if column == "TARGET":
        return "int"
    if column in DOUBLE_COLUMNS:
        return "double"
    if column.startswith(INT_PREFIXES):
        return "int"
    return "double"


@dataclass(frozen=True)
class Rule:
    """A row-level validity rule. `violation_sql` is TRUE when the row is invalid."""
    name: str
    violation_sql: str
    description: str


@dataclass(frozen=True)
class TableSpec:
    name: str                       # silver table name
    file: str                       # raw CSV file name
    grain: str                      # what one row represents
    primary_key: tuple[str, ...]    # () when the table has no natural key
    required_columns: tuple[str, ...]
    foreign_keys: tuple[tuple[str, str, str], ...] = ()   # (column, parent_table, parent_column)
    rules: tuple[Rule, ...] = ()
    placeholder_null_columns: tuple[str, ...] = ()        # 365243 -> NULL
    category_null_values: dict[str, tuple[str, ...]] = field(default_factory=dict)  # 'XNA' -> NULL
    max_reject_rate: float = 0.01
    expected_rows: int | None = None
    description: str = ""


def _not_null(*cols: str) -> tuple[Rule, ...]:
    return tuple(Rule(f"null_key:{c}", f"`{c}` IS NULL", f"{c} must be present") for c in cols)


_APP_REQUIRED = (
    "SK_ID_CURR", "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_LAST_PHONE_CHANGE",
)
_APP_RULES = (
    Rule("non_positive_income", "`AMT_INCOME_TOTAL` IS NULL OR `AMT_INCOME_TOTAL` <= 0",
         "Income must be present and > 0"),
    Rule("non_positive_credit", "`AMT_CREDIT` IS NULL OR `AMT_CREDIT` <= 0",
         "Credit amount must be present and > 0"),
    Rule("invalid_days_birth", "`DAYS_BIRTH` IS NULL OR `DAYS_BIRTH` >= 0",
         "DAYS_BIRTH is days before application, so it must be negative"),
)
_APP_CATEGORY_NULLS = {"CODE_GENDER": ("XNA",), "ORGANIZATION_TYPE": ("XNA",),
                       "NAME_FAMILY_STATUS": ("Unknown",)}

TABLES: dict[str, TableSpec] = {t.name: t for t in [
    TableSpec(
        name="application_train", file="application_train.csv",
        grain="One loan application (labelled)", primary_key=("SK_ID_CURR",),
        required_columns=("SK_ID_CURR", "TARGET") + _APP_REQUIRED[1:],
        rules=_not_null("SK_ID_CURR") + (
            Rule("invalid_target", "`TARGET` IS NULL OR `TARGET` NOT IN (0, 1)", "TARGET must be 0 or 1"),
        ) + _APP_RULES,
        placeholder_null_columns=("DAYS_EMPLOYED",),
        category_null_values=_APP_CATEGORY_NULLS,
        expected_rows=307_511,
        description="Main application table with the TARGET label (1 = payment difficulties).",
    ),
    TableSpec(
        name="application_test", file="application_test.csv",
        grain="One loan application (unlabelled)", primary_key=("SK_ID_CURR",),
        required_columns=_APP_REQUIRED,
        rules=_not_null("SK_ID_CURR") + _APP_RULES,
        placeholder_null_columns=("DAYS_EMPLOYED",),
        category_null_values=_APP_CATEGORY_NULLS,
        expected_rows=48_744,
        description="Application table without TARGET (Kaggle test set; not used for evaluation).",
    ),
    TableSpec(
        name="bureau", file="bureau.csv",
        grain="One credit at another institution reported to the Credit Bureau",
        primary_key=("SK_ID_BUREAU",),
        required_columns=("SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT",
                          "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
                          "AMT_CREDIT_SUM_OVERDUE", "CREDIT_TYPE"),
        foreign_keys=(("SK_ID_CURR", "application_*", "SK_ID_CURR"),),
        rules=_not_null("SK_ID_CURR", "SK_ID_BUREAU") + (
            Rule("negative_credit_sum", "`AMT_CREDIT_SUM` < 0", "AMT_CREDIT_SUM cannot be negative"),
        ),
        expected_rows=1_716_428,
        description="Previous credits of the applicant at other financial institutions.",
    ),
    TableSpec(
        name="bureau_balance", file="bureau_balance.csv",
        grain="One month of one Credit Bureau credit", primary_key=("SK_ID_BUREAU", "MONTHS_BALANCE"),
        required_columns=("SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"),
        foreign_keys=(("SK_ID_BUREAU", "bureau", "SK_ID_BUREAU"),),
        rules=_not_null("SK_ID_BUREAU", "MONTHS_BALANCE") + (
            Rule("invalid_status", "`STATUS` IS NULL OR `STATUS` NOT IN ('C','X','0','1','2','3','4','5')",
                 "STATUS must be C, X or 0-5"),
            Rule("future_month", "`MONTHS_BALANCE` > 0", "MONTHS_BALANCE is relative to application, <= 0"),
        ),
        expected_rows=27_299_925,
        description="Monthly status (days past due bucket) of Credit Bureau credits.",
    ),
    TableSpec(
        name="previous_application", file="previous_application.csv",
        grain="One earlier Home Credit application", primary_key=("SK_ID_PREV",),
        required_columns=("SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_APPLICATION",
                          "AMT_CREDIT", "NAME_CONTRACT_STATUS", "DAYS_DECISION"),
        foreign_keys=(("SK_ID_CURR", "application_*", "SK_ID_CURR"),),
        rules=_not_null("SK_ID_PREV", "SK_ID_CURR"),
        placeholder_null_columns=("DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
                                  "DAYS_LAST_DUE", "DAYS_TERMINATION"),
        category_null_values={"NAME_CASH_LOAN_PURPOSE": ("XNA", "XAP"), "CODE_REJECT_REASON": ("XNA",)},
        expected_rows=1_670_214,
        description="All previous Home Credit applications of applicants in the sample.",
    ),
    TableSpec(
        name="installments_payments", file="installments_payments.csv",
        grain="One payment (or missed payment) against one installment of a previous credit",
        primary_key=(),
        required_columns=("SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT",
                          "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"),
        foreign_keys=(("SK_ID_CURR", "application_*", "SK_ID_CURR"),
                      ("SK_ID_PREV", "previous_application", "SK_ID_PREV")),
        rules=_not_null("SK_ID_PREV", "SK_ID_CURR") + (
            Rule("negative_amount", "`AMT_PAYMENT` < 0 OR `AMT_INSTALMENT` < 0",
                 "Installment and payment amounts cannot be negative"),
        ),
        expected_rows=13_605_401,
        description="Repayment history of previous Home Credit credits.",
    ),
    TableSpec(
        name="pos_cash_balance", file="POS_CASH_balance.csv",
        grain="One month of one previous POS or cash loan", primary_key=(),
        required_columns=("SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "NAME_CONTRACT_STATUS",
                          "SK_DPD", "SK_DPD_DEF"),
        foreign_keys=(("SK_ID_CURR", "application_*", "SK_ID_CURR"),
                      ("SK_ID_PREV", "previous_application", "SK_ID_PREV")),
        rules=_not_null("SK_ID_PREV", "SK_ID_CURR") + (
            Rule("future_month", "`MONTHS_BALANCE` > 0", "MONTHS_BALANCE is relative to application, <= 0"),
            Rule("negative_dpd", "`SK_DPD` < 0 OR `SK_DPD_DEF` < 0", "Days past due cannot be negative"),
        ),
        expected_rows=10_001_358,
        description="Monthly balance snapshots of previous point-of-sale and cash loans.",
    ),
    TableSpec(
        name="credit_card_balance", file="credit_card_balance.csv",
        grain="One month of one previous credit card", primary_key=(),
        required_columns=("SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
                          "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"),
        foreign_keys=(("SK_ID_CURR", "application_*", "SK_ID_CURR"),
                      ("SK_ID_PREV", "previous_application", "SK_ID_PREV")),
        rules=_not_null("SK_ID_PREV", "SK_ID_CURR") + (
            Rule("future_month", "`MONTHS_BALANCE` > 0", "MONTHS_BALANCE is relative to application, <= 0"),
        ),
        expected_rows=3_840_312,
        description="Monthly balance snapshots of previous Home Credit credit cards.",
    ),
]}

# Bronze metadata columns added on load (never typed or validated as business data).
META_COLUMNS = ("_source_file", "_source_version", "_source_sha256", "_ingest_run_id",
                "_bronze_row_id", "_corrupt_record")
