"""Regenerate docs/feature_spec.md from hc_pipeline.features.FEATURES.

    python scripts/export_feature_spec.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hc_pipeline.features import (FEATURE_VERSION, FEATURES, GROUPS, SPLIT_SEED,  # noqa: E402
                                  TEST_FRACTION)

HEADER = f"""# Feature specification - {FEATURE_VERSION}

Generated from [`src/hc_pipeline/features.py`](../src/hc_pipeline/features.py) by
`scripts/export_feature_spec.py`. A unit test checks that the output table has exactly
these columns and types.

**Output:** `home_credit.gold.features_{FEATURE_VERSION}`, one row per `SK_ID_CURR` for train
and test applications (`DATASET` = train/test, `TARGET` NULL for test), plus
`gold.split_{FEATURE_VERSION}` and `gold.feature_registry` (this table, as data). The table
properties record `hc.feature_version`, `hc.source_version`, `hc.git_commit` and the
Delta version of every silver input (`hc.silver_versions`).

## Why these features

The hypothesis compares a model that uses only the **application form** with one that
also uses the applicant's **borrowing history**. The set therefore has two groups:

* **application** ({len(GROUPS['application'])} features): the baseline inputs. A few
  affordability ratios that lenders use (credit/income, annuity/income, credit term), age and
  employment length in years, and the external scores. Raw categoricals stay as strings and
  are encoded in the Phase-2 training pipeline.
* **history** ({len(GROUPS['history'])} features): one short block per history table. Each
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
* Features are kept to {len(FEATURES)} on purpose: each can be explained, and Phase 2
  can add more and measure their effect. No feature selection was applied in v1.

## Split

`gold.split_{FEATURE_VERSION}`: stratified {1 - TEST_FRACTION:.0%}/{TEST_FRACTION:.0%} split of
`application_train`. Inside each TARGET class, rows are ordered by
`xxhash64(SK_ID_CURR, {SPLIT_SEED})` and the first {TEST_FRACTION:.0%} go to `test`. The split
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
"""


def main() -> None:
    lines = [HEADER]
    for g in ("application", "history"):
        lines += [f"\n### {g.capitalize()} group\n",
                  "| Feature | Type | Source | Definition | Leakage / availability note |",
                  "|---|---|---|---|---|"]
        for f in FEATURES:
            if f.group == g:
                lines.append(f"| `{f.name}` | {f.dtype} | {f.source} | {f.definition} | {f.leakage_note} |")
    out = ROOT / "docs" / "feature_spec.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(FEATURES)} features)")


if __name__ == "__main__":
    main()
