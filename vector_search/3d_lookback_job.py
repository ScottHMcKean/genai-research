# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Embedding Lookback Job
# MAGIC
# MAGIC Validates embedding stability for each `registeredAccountId` by computing mean cosine similarity between the current month's vectors and up to 3 previous monthly snapshots.
# MAGIC
# MAGIC All comparison tables must share the same `id_column` + `vector_column` schema. Leave any missing snapshot widgets blank — they are filtered out automatically.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.removeAll()

dbutils.widgets.text("current_table",  "shm.genai.vs_w_vec_100k", "Current month table")
dbutils.widgets.text("snapshot_1",     "",                         "Month -3 (oldest)")
dbutils.widgets.text("snapshot_2",     "",                         "Month -2")
dbutils.widgets.text("snapshot_3",     "",                         "Month -1")
dbutils.widgets.text("id_column",      "registeredAccountId",     "ID column")
dbutils.widgets.text("vector_column",  "query_vec",               "Embedding column")
dbutils.widgets.text("output_table",   "",                         "Save results to (optional)")

# COMMAND ----------

# DBTITLE 1,Config and table list
CURRENT_TABLE = dbutils.widgets.get("current_table").strip()
SNAPSHOT_1    = dbutils.widgets.get("snapshot_1").strip()
SNAPSHOT_2    = dbutils.widgets.get("snapshot_2").strip()
SNAPSHOT_3    = dbutils.widgets.get("snapshot_3").strip()
ID_COLUMN     = dbutils.widgets.get("id_column").strip()
VECTOR_COLUMN = dbutils.widgets.get("vector_column").strip()
OUTPUT_TABLE  = dbutils.widgets.get("output_table").strip()

# Build ordered list oldest → newest, skipping empty widgets
_all = [t for t in [SNAPSHOT_1, SNAPSHOT_2, SNAPSHOT_3, CURRENT_TABLE] if t]
assert len(_all) >= 2, (
    "Need at least 2 tables to compare. "
    "Set current_table and at least one snapshot_N widget."
)

print(f"Snapshots ({len(_all)} total, oldest → newest):")
for i, t in enumerate(_all):
    age = len(_all) - 1 - i
    label = "current" if age == 0 else f"month -{age}"
    print(f"  snap_num={i}  [{label}]  {t}")

print()
print(f"ID column:      {ID_COLUMN}")
print(f"Vector column:  {VECTOR_COLUMN}")
if OUTPUT_TABLE:
    print(f"Output table:   {OUTPUT_TABLE}")
else:
    print("Output table:   (not saving — set output_table widget to persist results)")

# COMMAND ----------

# DBTITLE 1,Cosine similarity across snapshots
# Each snapshot gets an integer snap_num (0 = oldest, N-1 = current).
# Self-join pairs every row with its up to 3 prior snapshots for the same account.
# zip_with + aggregate implement dot product and L2 norms in pure Spark SQL.
# array<float> elements are cast to DOUBLE to avoid float32 precision loss.

_union = "\nUNION ALL\n  ".join(
    f"SELECT `{ID_COLUMN}` AS account_id, `{VECTOR_COLUMN}` AS vec, {i} AS snap_num"
    f" FROM {tbl}"
    for i, tbl in enumerate(_all)
)

stability_df = spark.sql(f"""
WITH snapshots AS (
  {_union}
),
pairs AS (
  SELECT
    a.account_id,
    a.snap_num                                                             AS snap,
    b.snap_num                                                             AS prev_snap,
    aggregate(
      zip_with(a.vec, b.vec, (x, y) -> CAST(x AS DOUBLE) * CAST(y AS DOUBLE)),
      0.0D, (acc, v) -> acc + v
    ) / NULLIF(
      sqrt(aggregate(transform(a.vec, x -> CAST(x AS DOUBLE) * CAST(x AS DOUBLE)),
           0.0D, (acc, v) -> acc + v)) *
      sqrt(aggregate(transform(b.vec, x -> CAST(x AS DOUBLE) * CAST(x AS DOUBLE)),
           0.0D, (acc, v) -> acc + v)),
      0.0
    )                                                                      AS cosine_sim
  FROM snapshots a
  JOIN snapshots b
    ON  a.account_id = b.account_id
    AND b.snap_num   < a.snap_num
    AND b.snap_num  >= a.snap_num - 3
)
SELECT
  account_id,
  COUNT(*)                                AS num_pairs,
  ROUND(AVG(cosine_sim),    6)            AS mean_cosine_sim,
  ROUND(MIN(cosine_sim),    6)            AS min_cosine_sim,
  ROUND(STDDEV(cosine_sim), 6)            AS stddev_cosine_sim
FROM pairs
GROUP BY account_id
ORDER BY mean_cosine_sim ASC
""")

# Cache so the summary query below doesn't recompute
stability_df.createOrReplaceTempView("_lookback_results")
display(stability_df)

# COMMAND ----------

# DBTITLE 1,Global distribution summary
summary = spark.sql("""
SELECT
  COUNT(*)                                           AS num_accounts,
  ROUND(AVG(mean_cosine_sim),                    6)  AS avg_mean_sim,
  ROUND(PERCENTILE_APPROX(mean_cosine_sim, 0.05), 6) AS p05,
  ROUND(PERCENTILE_APPROX(mean_cosine_sim, 0.25), 6) AS p25,
  ROUND(PERCENTILE_APPROX(mean_cosine_sim, 0.50), 6) AS p50,
  ROUND(PERCENTILE_APPROX(mean_cosine_sim, 0.75), 6) AS p75,
  ROUND(PERCENTILE_APPROX(mean_cosine_sim, 0.95), 6) AS p95,
  ROUND(MIN(mean_cosine_sim),                    6)  AS min_sim,
  ROUND(MAX(mean_cosine_sim),                    6)  AS max_sim,
  SUM(CASE WHEN mean_cosine_sim < 0.80 THEN 1 ELSE 0 END) AS unstable_below_0_80,
  SUM(CASE WHEN mean_cosine_sim < 0.90 THEN 1 ELSE 0 END) AS unstable_below_0_90,
  SUM(CASE WHEN mean_cosine_sim < 0.95 THEN 1 ELSE 0 END) AS unstable_below_0_95
FROM _lookback_results
""")
display(summary)

# COMMAND ----------

# DBTITLE 1,Save results (optional)
if OUTPUT_TABLE:
    (
        stability_df
            .write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(OUTPUT_TABLE)
    )
    print(f"Saved {stability_df.count():,} rows to {OUTPUT_TABLE}")
else:
    print("Skipping save — set the output_table widget to persist results.")
