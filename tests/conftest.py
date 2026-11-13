import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    s = (SparkSession.builder.master("local[2]").appName("hc-tests")
         .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "2")
         .config("spark.sql.ansi.enabled", "true")   # same default as recent Databricks runtimes
         .getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


@pytest.fixture
def csv_file(tmp_path):
    """Write text to a CSV file and return its path."""
    def _write(name: str, text: str) -> str:
        p = tmp_path / name
        p.write_text(text)
        return str(p)
    return _write
