"""Unit tests for bronze -> silver and the quality gate (tiny hand-written inputs)."""
import json

import pytest

from hc_pipeline import quality, transforms
from hc_pipeline.spec import TABLES, resolve_type

APP_HEADER = ("SK_ID_CURR,TARGET,NAME_CONTRACT_TYPE,CODE_GENDER,FLAG_OWN_CAR,FLAG_OWN_REALTY,"
              "CNT_CHILDREN,AMT_INCOME_TOTAL,AMT_CREDIT,AMT_ANNUITY,AMT_GOODS_PRICE,NAME_INCOME_TYPE,"
              "NAME_EDUCATION_TYPE,NAME_FAMILY_STATUS,NAME_HOUSING_TYPE,DAYS_BIRTH,DAYS_EMPLOYED,"
              "DAYS_REGISTRATION,DAYS_ID_PUBLISH,OCCUPATION_TYPE,ORGANIZATION_TYPE,CNT_FAM_MEMBERS,"
              "REGION_RATING_CLIENT,EXT_SOURCE_1,EXT_SOURCE_2,EXT_SOURCE_3,DAYS_LAST_PHONE_CHANGE")


def app_row(i, target="0", income="100000.0", days_emp="-1000", gender="F", days_birth="-12000"):
    return (f"{i},{target},Cash loans,{gender},N,Y,0,{income},500000.0,25000.0,450000.0,Working,"
            f"Higher education,Married,House / apartment,{days_birth},{days_emp},-3000.0,-2000,"
            f"Laborers,XNA,2.0,2,,0.5,0.6,-100.0")


def run(spark, csv_file, rows, table="application_train"):
    path = csv_file("t.csv", APP_HEADER + "\n" + "\n".join(rows) + "\n")
    bronze = transforms.to_bronze(transforms.read_raw_csv(spark, path), "v_test", "sha", "r1")
    out = transforms.bronze_to_silver(bronze, TABLES[table], "r1")
    return bronze, out


def reasons(out):
    return {r["bronze_row_id"]: set(r["reasons"]) for r in out["rejected"].collect()}


def test_type_resolution():
    assert resolve_type("SK_ID_CURR") == "bigint"
    assert resolve_type("TARGET") == "int"
    assert resolve_type("DAYS_BIRTH") == "int"
    assert resolve_type("AMT_CREDIT") == "double"
    assert resolve_type("FLAG_OWN_CAR") == "string"          # Y/N, despite the FLAG_ prefix
    assert resolve_type("FLAG_DOCUMENT_3") == "double"
    assert resolve_type("NAME_CONTRACT_STATUS") == "string"


def test_clean_rows_pass_and_are_typed(spark, csv_file):
    _, out = run(spark, csv_file, [app_row(1), app_row(2, target="1")])
    acc = out["accepted"]
    assert acc.count() == 2 and out["rejected"].count() == 0
    types = dict(acc.dtypes)
    assert types["SK_ID_CURR"] == "bigint" and types["TARGET"] == "int"
    assert types["AMT_INCOME_TOTAL"] == "double" and types["DAYS_REGISTRATION"] == "int"
    assert acc.filter("EXT_SOURCE_1 IS NULL").count() == 2      # empty string -> NULL


def test_cleaning_rules_are_traceable(spark, csv_file):
    _, out = run(spark, csv_file, [app_row(1, days_emp="365243", gender="XNA")])
    r = out["accepted"].first()
    assert r["DAYS_EMPLOYED"] is None and r["DAYS_EMPLOYED_ANOM"] == 1
    assert r["CODE_GENDER"] is None and r["ORGANIZATION_TYPE"] is None
    assert set(r["_cleaned_cols"]) == {"DAYS_EMPLOYED", "CODE_GENDER", "ORGANIZATION_TYPE"}


def test_invalid_rows_go_to_quarantine_with_reasons(spark, csv_file):
    rows = [app_row(1), app_row(2, income="abc"), app_row(3, target="7"),
            app_row("", target="0"), app_row(5, days_birth="12.5"), app_row(6, income="-5")]
    bronze, out = run(spark, csv_file, rows)
    assert out["accepted"].count() == 1
    got = {json.loads(r["raw_record"]).get("SK_ID_CURR"): set(r["reasons"])
           for r in out["rejected"].collect()}
    assert got["2"] == {"type:AMT_INCOME_TOTAL", "non_positive_income"}
    assert got["3"] == {"invalid_target"}
    assert got[None] == {"null_key:SK_ID_CURR"}
    assert "type:DAYS_BIRTH" in got["5"]                       # non-integer day count
    assert got["6"] == {"non_positive_income"}


def test_duplicates(spark, csv_file):
    rows = [app_row(1), app_row(1),                             # exact duplicate -> keep one
            app_row(2), app_row(2, income="999.0")]             # key conflict -> reject both
    _, out = run(spark, csv_file, rows)
    assert [r["SK_ID_CURR"] for r in out["accepted"].collect()] == [1]
    rs = sorted(tuple(r["reasons"]) for r in out["rejected"].collect())
    assert rs == [("duplicate_key_conflict",), ("duplicate_key_conflict",), ("duplicate_row",)]


def test_malformed_line_is_quarantined(spark, csv_file):
    _, out = run(spark, csv_file, [app_row(1), "2,0,Cash loans,F,N"])   # too few fields
    assert out["accepted"].count() == 1
    assert "malformed_line" in out["rejected"].first()["reasons"]


def test_missing_required_column_is_reported(spark, csv_file):
    header = APP_HEADER.replace(",EXT_SOURCE_3", "")
    row = app_row(1).rsplit(",", 2)
    path = csv_file("m.csv", header + "\n" + row[0] + "," + row[2] + "\n")
    bronze = transforms.to_bronze(transforms.read_raw_csv(spark, path), "v", None, "r")
    out = transforms.bronze_to_silver(bronze, TABLES["application_train"], "r")
    assert out["missing_columns"] == ["EXT_SOURCE_3"]


def test_row_accounting_and_gate(spark, csv_file):
    rows = [app_row(i, target=str(int(i % 12 == 0))) for i in range(1, 101)] + [app_row(500, income="x"), app_row(501, income="x")]
    bronze, out = run(spark, csv_file, rows)
    spec = TABLES["application_train"]
    m = quality.table_metrics(spec, bronze.count(), out["accepted"], out["rejected"], [])
    assert m["rows_in"] == m["rows_accepted"] + m["rows_rejected"] == 102
    checks = {c.check: c for c in quality.evaluate_table(spec, m, enforce_expected_rows=False)}
    assert checks["accounting.rows_in_equals_accepted_plus_rejected"].passed
    assert checks["uniqueness.primary_key"].passed
    assert not checks["validity.reject_rate"].passed           # 2/102 = 1.96 % > 1 %
    with pytest.raises(quality.QualityGateError):
        quality.raise_if_blocked(list(checks.values()))
