"""Thin storage adapter.

* `UnityCatalogStore` (Databricks): Delta tables named catalog.schema.table, with
  table properties, comments and informational constraints.
* `LocalStore` (laptop / CI): Parquet tables in a local Spark warehouse named
  `<catalog>_<schema>.table`, so the whole job can be smoke-tested without Azure.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

SCHEMAS = ("bronze", "silver_staging", "silver", "gold", "ops")


class Store:
    """`single_schema`: when set (e.g. "default"), every layer lives in that one schema
    and the layer becomes a table-name prefix (bronze_x, silver_x, ...). Used when the
    user may create tables but not schemas in the catalog."""

    def __init__(self, spark: SparkSession, catalog: str, single_schema: str = ""):
        self.spark, self.catalog, self.single_schema = spark, catalog, single_schema

    def _parts(self, schema: str, table: str) -> tuple[str, str]:
        if self.single_schema:
            return self.single_schema, f"{schema}_{table}"
        return schema, table

    # naming ---------------------------------------------------------------
    def fq(self, schema: str, table: str) -> str:
        raise NotImplementedError

    def ensure_schemas(self) -> None:
        raise NotImplementedError

    # io -------------------------------------------------------------------
    def exists(self, schema: str, table: str) -> bool:
        return self.spark.catalog.tableExists(self.fq(schema, table))

    def read(self, schema: str, table: str) -> DataFrame:
        return self.spark.table(self.fq(schema, table))

    def overwrite(self, df: DataFrame, schema: str, table: str, props: dict | None = None) -> None:
        raise NotImplementedError

    def append(self, df: DataFrame, schema: str, table: str) -> None:
        df.write.mode("append").option("mergeSchema", "true").saveAsTable(self.fq(schema, table))

    def replace_partition(self, df: DataFrame, schema: str, table: str, where: str) -> None:
        """Idempotent append: delete rows matching `where`, then append `df`."""
        raise NotImplementedError

    def props(self, schema: str, table: str) -> dict:
        if not self.exists(schema, table):
            return {}
        rows = self.spark.sql(f"SHOW TBLPROPERTIES {self.fq(schema, table)}").collect()
        return {r["key"]: r["value"] for r in rows}

    def set_props(self, schema: str, table: str, props: dict) -> None:
        if props:
            kv = ", ".join(f"'{k}' = '{str(v)}'" for k, v in props.items())
            self.spark.sql(f"ALTER TABLE {self.fq(schema, table)} SET TBLPROPERTIES ({kv})")

    def current_version(self, schema: str, table: str) -> str | None:
        return None

    def apply_metadata(self, schema: str, table: str, comment: str,
                       column_comments: dict[str, str], primary_key: tuple[str, ...]) -> list[str]:
        return []


class UnityCatalogStore(Store):
    def fq(self, schema: str, table: str) -> str:
        s, t = self._parts(schema, table)
        return f"`{self.catalog}`.`{s}`.`{t}`"

    def ensure_schemas(self) -> None:
        if self.single_schema:
            return
        for s in SCHEMAS:
            self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{self.catalog}`.`{s}`")

    def overwrite(self, df, schema, table, props=None):
        (df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
         .saveAsTable(self.fq(schema, table)))
        self.set_props(schema, table, props or {})

    def replace_partition(self, df, schema, table, where):
        if self.exists(schema, table):
            self.spark.sql(f"DELETE FROM {self.fq(schema, table)} WHERE {where}")
        self.append(df, schema, table)

    def current_version(self, schema, table):
        row = self.spark.sql(f"DESCRIBE HISTORY {self.fq(schema, table)} LIMIT 1").first()
        return str(row["version"]) if row else None

    def apply_metadata(self, schema, table, comment, column_comments, primary_key):
        """Table/column comments and an informational PRIMARY KEY. Returns warnings
        instead of failing, because constraints need Unity Catalog permissions."""
        name, warnings = self.fq(schema, table), []
        table = self._parts(schema, table)[1]
        esc = lambda s: s.replace("'", "\\'")  # noqa: E731
        try:
            self.spark.sql(f"COMMENT ON TABLE {name} IS '{esc(comment)}'")
            cols = set(self.spark.table(name).columns)
            for c, text in column_comments.items():
                if c in cols:
                    self.spark.sql(f"ALTER TABLE {name} ALTER COLUMN `{c}` COMMENT '{esc(text)}'")
            if primary_key:
                for c in primary_key:
                    self.spark.sql(f"ALTER TABLE {name} ALTER COLUMN `{c}` SET NOT NULL")
                cols_sql = ", ".join(f"`{c}`" for c in primary_key)
                self.spark.sql(f"ALTER TABLE {name} ADD CONSTRAINT `pk_{table}` PRIMARY KEY ({cols_sql})")
        except Exception as e:  # noqa: BLE001 - metadata must never block publication
            warnings.append(f"{table}: metadata not fully applied ({type(e).__name__}: {str(e)[:200]})")
        return warnings


class LocalStore(Store):
    def fq(self, schema: str, table: str) -> str:
        s, t = self._parts(schema, table)
        return f"{self.catalog}_{s}.{t}"

    def ensure_schemas(self) -> None:
        for s in ([self.single_schema] if self.single_schema else SCHEMAS):
            self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {self.catalog}_{s}")

    def overwrite(self, df, schema, table, props=None):
        name = self.fq(schema, table)
        tmp = f"{name}__tmp"
        # Parquet cannot overwrite a table it is reading from: stage through a temp table.
        df.write.mode("overwrite").format("parquet").saveAsTable(tmp)
        self.spark.sql(f"DROP TABLE IF EXISTS {name}")
        self.spark.table(tmp).write.mode("overwrite").format("parquet").saveAsTable(name)
        self.spark.sql(f"DROP TABLE IF EXISTS {tmp}")
        self.set_props(schema, table, props or {})

    def replace_partition(self, df, schema, table, where):
        name = self.fq(schema, table)
        if self.exists(schema, table):
            keep = self.spark.table(name).filter(f"NOT ({where})")
            self.overwrite(keep.unionByName(df, allowMissingColumns=True), schema, table)
        else:
            df.write.format("parquet").saveAsTable(name)


def make_store(spark: SparkSession, catalog: str, local: bool, single_schema: str = "") -> Store:
    return (LocalStore if local else UnityCatalogStore)(spark, catalog, single_schema)
