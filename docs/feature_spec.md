# Feature specification - v1

Generated from [`src/hc_pipeline/features.py`](../src/hc_pipeline/features.py) by
`scripts/export_feature_spec.py`. A unit test checks that the output table has exactly
these columns and types.

**Output:** `home_credit.gold.features_v1`, one row per `SK_ID_CURR` for train
and test applications (`DATASET` = train/test, `TARGET` NULL for test), plus
`gold.split_v1` and `gold.feature_registry` (this table, as data). The table
properties record `hc.feature_version`, `hc.source_version`, `hc.git_commit` and the
Delta version of every silver input (`hc.silver_versions`).

## Why these features

The hypothesis compares a model that uses only the **application form** with one that
also uses the applicant's **borrowing history**. The set therefore has two groups:

* **application** (24 features): the baseline inputs. A few
  affordability ratios that lenders use (credit/income, annuity/income, credit term), age and
  employment length in years, and the external scores. Raw categoricals stay as strings and
  are encoded in the Phase-2 training pipeline.
* **history** (26 features): one short block per history table. Each
  block covers three questions: *how much* history (counts), *how indebted* (debt and
  utilisation ratios) and *how well it was repaid* (late and underpaid rates, days past
  due). The repayment-behaviour features are the ones the hypothesis is about.

Design choices:

* **Counts are 0 when there is no history; rates and ratios stay NULL.** "No bureau
  credits" is a real 0, but a late-payment rate over zero payments is undefined. LightGBM
  handles NULL natively. For logistic regression, the Phase-2 pipeline imputes on the
  training split only.
* **Installments are aggregated per installment first**, then per applicant, because
  one installment can be paid in several rows. Counting rows would turn a split payment
  into two "underpaid" installments.
* **Bureau balance** counts only months with a days-past-due status (0-5). `C` (closed)
  and `X` (unknown) carry no repayment information.
* Features are kept to 50 on purpose: each can be explained, and Phase 2
  can add more and measure their effect. No feature selection was applied in v1.

## Split

`gold.split_v1`: stratified 80%/20% split of
`application_train`. Inside each TARGET class, rows are ordered by
`xxhash64(SK_ID_CURR, 3202)` and the first 20% go to `test`. The split
is deterministic (same input gives the same split on any cluster) and both models in the
hypothesis use it. `application_test` (Kaggle's unlabelled set) is not used for evaluation.

## Leakage risks and how they are handled

| Risk | Handling |
|---|---|
| **Future history records.** A history row dated after the application would reveal what happened next. | All history is filtered to relative day/month **≤ 0** before aggregation (`DAYS_CREDIT`, `DAYS_DECISION`, `DAYS_INSTALMENT`, `MONTHS_BALANCE`). A payment date after the application is ignored (`DAYS_ENTRY_PAYMENT > 0`). Silver also rejects `MONTHS_BALANCE > 0`. A unit test feeds a future-dated bureau credit and checks that it is excluded. |
| **Target in features.** | `TARGET` is an ID column, not a feature. No target encoding is used in v1. |
| **Fitting on the test split.** Imputation, scaling and encoding fitted on all rows leak test statistics. | v1 contains only row-wise values and per-applicant aggregates (no statistics across applicants), so the feature table can be computed before the split. All fitted transforms belong in the Phase-2 pipeline and are fit on `SPLIT = 'train'` only. |
| **Same applicant in train and test.** | The split is by `SK_ID_CURR` (one row per applicant). The gate blocks if Kaggle's train and test sets share IDs. |
| **EXT_SOURCE_1..3.** These are third-party scores of unknown timing. If computed after the loan, they would leak. | Kept, because they are part of the application record. Kaggle only describes them as "normalized score from external data source", so the timing is unverified. Phase 2 should report results with and without them. |
| **`CODE_GENDER`.** Not leakage but a protected attribute. | Kept for analysis; fairness review before any production use. |
| **Bureau `DAYS_CREDIT_ENDDATE` > 0.** | Not used in v1. It is a planned end date (known at application), so it would be safe, but it is excluded to keep v1 small. |

## Feature table

Availability: all features below can be computed when an application arrives,
from the form plus the history tables as of the application date.


### Application group

| Feature | Type | Source | Definition | Leakage / availability note |
|---|---|---|---|---|
| `AMT_INCOME_TOTAL` | double | application | Declared annual income. | Known at application time. |
| `AMT_CREDIT` | double | application | Credit amount requested. | Known at application time. |
| `AMT_ANNUITY` | double | application | Loan annuity (periodic payment). | Known at application time. |
| `AMT_GOODS_PRICE` | double | application | Price of goods financed. | Known at application time. |
| `CREDIT_INCOME_RATIO` | double | application | AMT_CREDIT / AMT_INCOME_TOTAL. | Known at application time. |
| `ANNUITY_INCOME_RATIO` | double | application | AMT_ANNUITY / AMT_INCOME_TOTAL (payment burden). | Known at application time. |
| `CREDIT_TERM` | double | application | AMT_CREDIT / AMT_ANNUITY (approximate number of payments). | Known at application time. |
| `GOODS_CREDIT_RATIO` | double | application | AMT_GOODS_PRICE / AMT_CREDIT (share of credit covering goods). | Known at application time. |
| `AGE_YEARS` | double | application | -DAYS_BIRTH / 365.25. | Known at application time. |
| `EMPLOYED_YEARS` | double | application | -DAYS_EMPLOYED / 365.25; NULL when DAYS_EMPLOYED was the 365243 placeholder. | Known at application time. |
| `DAYS_EMPLOYED_ANOM` | int | application | 1 if DAYS_EMPLOYED was the 365243 placeholder (mostly pensioners), else 0. | Known at application time. |
| `EXT_SOURCE_1` | double | application | External score 1 (normalised). | Third-party score; provenance unknown. Check it is computed before the application. |
| `EXT_SOURCE_2` | double | application | External score 2 (normalised). | Third-party score; provenance unknown. Check it is computed before the application. |
| `EXT_SOURCE_3` | double | application | External score 3 (normalised). | Third-party score; provenance unknown. Check it is computed before the application. |
| `EXT_SOURCE_MEAN` | double | application | Mean of the available EXT_SOURCE_1..3 (NULL if none). | Same as EXT_SOURCE_*. |
| `CNT_CHILDREN` | double | application | Number of children. | Known at application time. |
| `CNT_FAM_MEMBERS` | double | application | Family size. | Known at application time. |
| `REGION_RATING_CLIENT` | double | application | Home Credit region rating (1-3). | Known at application time. |
| `NAME_CONTRACT_TYPE` | string | application | Cash or revolving loan. | Known at application time. |
| `CODE_GENDER` | string | application | Gender ('XNA' -> NULL in silver). | Protected attribute: kept for analysis; fairness review required before use in Phase 2. |
| `NAME_INCOME_TYPE` | string | application | Income type. | Known at application time. |
| `NAME_EDUCATION_TYPE` | string | application | Highest education. | Known at application time. |
| `FLAG_OWN_CAR` | string | application | Owns a car (Y/N). | Known at application time. |
| `FLAG_OWN_REALTY` | string | application | Owns real estate (Y/N). | Known at application time. |

### History group

| Feature | Type | Source | Definition | Leakage / availability note |
|---|---|---|---|---|
| `BUR_LOAN_COUNT` | int | bureau | Number of Credit Bureau credits (0 if none). | Known at application time. |
| `BUR_ACTIVE_COUNT` | int | bureau | Credits with CREDIT_ACTIVE = 'Active'. | Known at application time. |
| `BUR_ACTIVE_DEBT_SUM` | double | bureau | Sum of AMT_CREDIT_SUM_DEBT over active credits. | Known at application time. |
| `BUR_DEBT_CREDIT_RATIO` | double | bureau | Active debt / active credit amount (NULL without active credit). | Known at application time. |
| `BUR_DEBT_INCOME_RATIO` | double | bureau + application | BUR_ACTIVE_DEBT_SUM / AMT_INCOME_TOTAL. | Known at application time. |
| `BUR_OVERDUE_SUM` | double | bureau | Sum of AMT_CREDIT_SUM_OVERDUE. | Known at application time. |
| `BUR_MAX_DAY_OVERDUE` | double | bureau | Max CREDIT_DAY_OVERDUE. | Known at application time. |
| `BUR_DAYS_SINCE_LAST_CREDIT` | double | bureau | -max(DAYS_CREDIT): days since the most recent bureau credit was opened. | Known at application time. |
| `BB_MONTHS_COUNT` | int | bureau_balance | Months of bureau balance history with a DPD status (0-5). | Known at application time. |
| `BB_LATE_MONTH_RATE` | double | bureau_balance | Share of those months with STATUS 1-5 (any days past due). | Known at application time. |
| `PREV_APP_COUNT` | int | previous_application | Number of previous applications. | Known at application time. |
| `PREV_REFUSED_COUNT` | int | previous_application | Previous applications with NAME_CONTRACT_STATUS = 'Refused'. | Known at application time. |
| `PREV_APPROVED_RATE` | double | previous_application | Approved / all previous applications. | Known at application time. |
| `PREV_CREDIT_APPLICATION_RATIO_MEAN` | double | previous_application | Mean of AMT_CREDIT / AMT_APPLICATION (granted vs. asked). | Known at application time. |
| `PREV_DAYS_SINCE_LAST_DECISION` | double | previous_application | -max(DAYS_DECISION). | Known at application time. |
| `INST_COUNT` | int | installments_payments | Number of past installments (payments grouped per installment). | Known at application time. |
| `INST_LATE_RATE` | double | installments_payments | Share of installments whose last payment came after the due day. | Known at application time. |
| `INST_DPD_MEAN` | double | installments_payments | Mean days past due per installment (0 when on time). | Known at application time. |
| `INST_UNDERPAID_RATE` | double | installments_payments | Share of installments where total paid < amount due. | Known at application time. |
| `INST_PAID_RATIO` | double | installments_payments | Total paid / total due. | Known at application time. |
| `POS_MONTHS_COUNT` | int | POS_CASH_balance | Months of POS/cash loan history. | Known at application time. |
| `POS_LATE_MONTH_RATE` | double | POS_CASH_balance | Share of months with SK_DPD > 0. | Known at application time. |
| `POS_DPD_MAX` | double | POS_CASH_balance | Max SK_DPD. | Known at application time. |
| `CC_MONTHS_COUNT` | int | credit_card_balance | Months of credit card history. | Known at application time. |
| `CC_UTILIZATION_MEAN` | double | credit_card_balance | Mean of AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL (months with limit > 0). | Known at application time. |
| `CC_DPD_MAX` | double | credit_card_balance | Max SK_DPD on credit cards. | Known at application time. |
