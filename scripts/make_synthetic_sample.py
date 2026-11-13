"""Generate a tiny SYNTHETIC copy of the Home Credit tables for tests and smoke runs.

The values are random; only the file names, column names and data quirks (365243
placeholder, 'XNA' categories, missing values, duplicates) mimic the real dataset.
Never use this output for analysis.

    python scripts/make_synthetic_sample.py --out data/synthetic --n-apps 2000
    python scripts/make_synthetic_sample.py --out data/synthetic_bad --inject-errors
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n_apps: int, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(100002, 100002 + n_apps)
    n_train = int(n_apps * 0.86)

    def app(ids_, labelled):
        n = len(ids_)
        income = rng.lognormal(11.9, 0.5, n).round(1)
        credit = (income * rng.uniform(1, 6, n)).round(1)
        annuity = (credit / rng.uniform(10, 40, n)).round(1)
        days_emp = -rng.integers(30, 12000, n)
        days_emp[rng.random(n) < 0.18] = 365243
        ext = lambda p: np.where(rng.random(n) < p, np.nan, rng.uniform(0, 1, n).round(6))
        e2 = ext(0.002)
        df = pd.DataFrame({
            "SK_ID_CURR": ids_,
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[.9, .1]),
            "CODE_GENDER": rng.choice(["F", "M", "XNA"], n, p=[.65, .3495, .0005]),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n), "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
            "CNT_CHILDREN": rng.choice([0, 1, 2, 3], n, p=[.7, .2, .08, .02]),
            "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": np.where(rng.random(n) < .001, np.nan, (credit * .9).round(1)),
            "NAME_TYPE_SUITE": rng.choice(["Unaccompanied", "Family", None], n, p=[.8, .15, .05]),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "Commercial associate"], n),
            "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education"], n),
            "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Unknown"], n,
                                             p=[.7, .2995, .0005]),
            "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents"], n),
            "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.07, n).round(6),
            "DAYS_BIRTH": -rng.integers(7500, 25200, n), "DAYS_EMPLOYED": days_emp,
            "DAYS_REGISTRATION": (-rng.integers(0, 20000, n)).astype(float),
            "DAYS_ID_PUBLISH": -rng.integers(0, 7000, n),
            "OWN_CAR_AGE": np.where(rng.random(n) < .65, np.nan, rng.integers(0, 30, n)),
            "OCCUPATION_TYPE": rng.choice(["Laborers", "Sales staff", None], n, p=[.4, .3, .3]),
            "CNT_FAM_MEMBERS": rng.integers(1, 5, n).astype(float),
            "REGION_RATING_CLIENT": rng.choice([1, 2, 3], n),
            "ORGANIZATION_TYPE": rng.choice(["Business Entity Type 3", "XNA", "Self-employed"], n),
            "EXT_SOURCE_1": ext(0.56), "EXT_SOURCE_2": e2, "EXT_SOURCE_3": ext(0.2),
            "APARTMENTS_AVG": ext(0.5), "FONDKAPREMONT_MODE": rng.choice(["reg oper account", None], n),
            "DAYS_LAST_PHONE_CHANGE": (-rng.integers(0, 4000, n)).astype(float),
            "FLAG_DOCUMENT_3": rng.choice([0, 1], n),
            "AMT_REQ_CREDIT_BUREAU_YEAR": np.where(rng.random(n) < .13, np.nan, rng.integers(0, 6, n)),
        })
        if labelled:
            risk = 1 / (1 + np.exp(-(-2.6 + 2.0 * (0.5 - np.nan_to_num(e2, nan=.5)))))
            df.insert(1, "TARGET", (rng.random(n) < risk).astype(int))
        return df

    train, test = app(ids[:n_train], True), app(ids[n_train:], False)

    nb = n_apps * 5
    bureau = pd.DataFrame({
        "SK_ID_CURR": rng.choice(ids, nb), "SK_ID_BUREAU": np.arange(5000000, 5000000 + nb),
        "CREDIT_ACTIVE": rng.choice(["Closed", "Active", "Sold"], nb, p=[.62, .37, .01]),
        "CREDIT_CURRENCY": "currency 1", "DAYS_CREDIT": -rng.integers(0, 2900, nb),
        "CREDIT_DAY_OVERDUE": np.where(rng.random(nb) < .97, 0, rng.integers(1, 2000, nb)),
        "DAYS_CREDIT_ENDDATE": rng.integers(-3000, 30000, nb).astype(float),
        "DAYS_ENDDATE_FACT": np.where(rng.random(nb) < .37, np.nan, -rng.integers(0, 2900, nb)),
        "AMT_CREDIT_MAX_OVERDUE": np.where(rng.random(nb) < .65, np.nan, rng.uniform(0, 5000, nb).round(2)),
        "CNT_CREDIT_PROLONG": 0,
        "AMT_CREDIT_SUM": rng.lognormal(12, 1, nb).round(2),
        "AMT_CREDIT_SUM_DEBT": np.where(rng.random(nb) < .15, np.nan, rng.lognormal(10, 2, nb).round(2)),
        "AMT_CREDIT_SUM_LIMIT": np.nan, "AMT_CREDIT_SUM_OVERDUE": np.where(rng.random(nb) < .99, 0.0, 500.0),
        "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan"], nb),
        "DAYS_CREDIT_UPDATE": -rng.integers(0, 2900, nb), "AMT_ANNUITY": np.nan,
    })
    bb_ids = rng.choice(bureau.SK_ID_BUREAU, nb // 2, replace=False)
    bb = pd.DataFrame([(b, -m, rng.choice(list("C0X1"), p=[.5, .3, .15, .05]))
                       for b in bb_ids for m in range(rng.integers(1, 12))],
                      columns=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"])

    npv = n_apps * 4
    prev = pd.DataFrame({
        "SK_ID_PREV": np.arange(1000000, 1000000 + npv), "SK_ID_CURR": rng.choice(ids, npv),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Consumer loans", "Revolving loans"], npv),
        "AMT_ANNUITY": rng.uniform(1000, 30000, npv).round(2),
        "AMT_APPLICATION": rng.lognormal(11, 1, npv).round(2), "AMT_CREDIT": rng.lognormal(11, 1, npv).round(2),
        "FLAG_LAST_APPL_PER_CONTRACT": "Y",
        "NAME_CASH_LOAN_PURPOSE": rng.choice(["XAP", "XNA", "Repairs"], npv),
        "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused", "Canceled", "Unused offer"], npv,
                                           p=[.62, .17, .19, .02]),
        "DAYS_DECISION": -rng.integers(1, 2900, npv), "CODE_REJECT_REASON": rng.choice(["XAP", "HC"], npv),
        "CNT_PAYMENT": rng.choice([6, 12, 24, 36], npv).astype(float),
        "DAYS_FIRST_DUE": np.where(rng.random(npv) < .4, 365243, -rng.integers(1, 2900, npv)).astype(float),
        "DAYS_TERMINATION": np.where(rng.random(npv) < .5, 365243, -rng.integers(1, 2900, npv)).astype(float),
    })
    approved = prev[prev.NAME_CONTRACT_STATUS == "Approved"]
    rows = []
    for r in approved.itertuples():
        for k in range(1, int(r.CNT_PAYMENT) // 2 + 1):
            due = r.DAYS_DECISION + 30 * k
            if due > -1:
                break
            late = rng.random() < .08
            rows.append((r.SK_ID_PREV, r.SK_ID_CURR, 1.0, k, float(due),
                         float(due + (rng.integers(1, 40) if late else -rng.integers(0, 10))),
                         r.AMT_ANNUITY, r.AMT_ANNUITY * (0.5 if rng.random() < .03 else 1)))
    inst = pd.DataFrame(rows, columns=["SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_VERSION",
                                       "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
                                       "AMT_INSTALMENT", "AMT_PAYMENT"])
    pos_src = approved[approved.NAME_CONTRACT_TYPE != "Revolving loans"]
    pos = pd.DataFrame([(r.SK_ID_PREV, r.SK_ID_CURR, -m, r.CNT_PAYMENT, max(r.CNT_PAYMENT - m, 0),
                         "Active", int(rng.random() < .03) * rng.integers(1, 30), 0)
                        for r in pos_src.itertuples() for m in range(1, 6)],
                       columns=["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT",
                                "CNT_INSTALMENT_FUTURE", "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"])
    cc_src = prev[prev.NAME_CONTRACT_TYPE == "Revolving loans"]
    cc = pd.DataFrame([(r.SK_ID_PREV, r.SK_ID_CURR, -m, rng.uniform(0, 90000), 90000.0,
                        rng.uniform(0, 5000), "Active", int(rng.random() < .02) * 5)
                       for r in cc_src.itertuples() for m in range(1, 6)],
                      columns=["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
                               "AMT_CREDIT_LIMIT_ACTUAL", "AMT_PAYMENT_CURRENT", "NAME_CONTRACT_STATUS",
                               "SK_DPD"])

    # A few exact duplicate rows, as real exports sometimes contain.
    bureau = pd.concat([bureau, bureau.head(3)], ignore_index=True)
    return {"application_train.csv": train, "application_test.csv": test, "bureau.csv": bureau,
            "bureau_balance.csv": bb, "previous_application.csv": prev,
            "installments_payments.csv": inst, "POS_CASH_balance.csv": pos,
            "credit_card_balance.csv": cc}


def inject_errors(tables: dict[str, pd.DataFrame]) -> None:
    """Corrupt application_train so the quality gate must fail."""
    t = tables["application_train.csv"].astype(object)
    n = len(t)
    t.loc[: n // 10, "AMT_INCOME_TOTAL"] = "not_a_number"   # ~10% type errors
    t.loc[n // 10: n // 10 + 20, "TARGET"] = 7              # invalid label
    t.loc[5:9, "SK_ID_CURR"] = None                          # missing keys
    tables["application_train.csv"] = pd.concat([t, t.tail(5)], ignore_index=True)  # duplicates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-apps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inject-errors", action="store_true")
    a = ap.parse_args()
    tables = make(a.n_apps, a.seed)
    if a.inject_errors:
        inject_errors(tables)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / name, index=False)
        print(f"{name:28s} {len(df):>8,d} rows")


if __name__ == "__main__":
    main()
