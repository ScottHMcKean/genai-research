# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Async Vector Search Inference Job
# MAGIC
# MAGIC Minimal notebook for high-throughput Vector Search inference with:
# MAGIC
# MAGIC | Feature | Detail |
# MAGIC |---|---|
# MAGIC | Parameters | `index_name`, `output_table`, `id_column`, `vector_column` |
# MAGIC | Fast writes | large append-only Delta checkpoints |
# MAGIC | Graceful backoff | reduces concurrency on 429 / 503 |
# MAGIC | Progress logging | prints rows, QPS, latency, and concurrency per batch |

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install --quiet aiohttp databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Cell 3
import asyncio
import time
from datetime import datetime, timezone

import aiohttp
import pandas as pd
from databricks.sdk import WorkspaceClient
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType, TimestampType

# ── Widgets ────────────────────────────────────────────────────────────
INDEX_NAME          = dbutils.widgets.get("index_name")
OUTPUT_TABLE        = dbutils.widgets.get("output_table")
ID_COLUMN           = dbutils.widgets.get("id_column")
VECTOR_COLUMN       = dbutils.widgets.get("vector_column")
INITIAL_CONCURRENCY = int(dbutils.widgets.get("initial_concurrency"))
TOP_K               = int(dbutils.widgets.get("top_k"))
BATCH_SIZE          = int(dbutils.widgets.get("batch_size"))
INFERENCE_MONTH     = dbutils.widgets.get("inference_month")
SOURCE_TABLE        = INDEX_NAME.removesuffix("_index")

assert OUTPUT_TABLE, "output_table widget must be set before running"

# ── Internal globals (not exposed as widgets) ────────────────────────────
MIN_CONCURRENCY    = 5      # floor — never go below this under backoff
MAX_RETRIES        = 5      # per-request retry attempts on transient errors
BACKOFF_THRESHOLD  = 0.05   # halve concurrency if >5% of batch is rate-limited
SCALE_UP_THRESHOLD = 0.01   # grow 25% if <1% is rate-limited; no ceiling

# ── Auth ──────────────────────────────────────────────────────────────────
w     = WorkspaceClient()
HOST  = w.config.host.rstrip("/")
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

print(f"Index:       {INDEX_NAME}")
print(f"Source:      {SOURCE_TABLE}")
print(f"Output:      {OUTPUT_TABLE}")
print(f"batch_size={BATCH_SIZE:,}  top_k={TOP_K}  initial_concurrency={INITIAL_CONCURRENCY}")

# COMMAND ----------

# DBTITLE 1,Cell 4
# Wide format: one row per query, neighbors collected into arrays
OUTPUT_SCHEMA = StructType([
    StructField(ID_COLUMN,             StringType(),              False),
    StructField("inference_month",     StringType(),              False),
    StructField("matched_account_ids", ArrayType(StringType()),   True),
    StructField("search_scores",       ArrayType(DoubleType()),   True),
    StructField("processed_at",        TimestampType(),           True),
])

if spark.catalog.tableExists(OUTPUT_TABLE):
    existing_cols = {f.name for f in spark.table(OUTPUT_TABLE).schema.fields}
    if ID_COLUMN in existing_cols:
        done_ids = set(
            spark.table(OUTPUT_TABLE)
            .select(ID_COLUMN)
            .distinct()
            .toPandas()[ID_COLUMN]
            .tolist()
        )
    else:
        # Table exists but has a different schema (e.g. old long format) — cannot resume
        done_ids = set()
    print(f"Resuming from {len(done_ids):,} completed query IDs")
else:
    spark.createDataFrame([], OUTPUT_SCHEMA).write.format("delta").mode("error").saveAsTable(OUTPUT_TABLE)
    done_ids = set()
    print(f"Created output table: {OUTPUT_TABLE}")

all_rows = spark.table(SOURCE_TABLE).select(ID_COLUMN, VECTOR_COLUMN).toPandas()
all_rows[ID_COLUMN] = all_rows[ID_COLUMN].astype(str)

if done_ids:
    all_rows = all_rows[~all_rows[ID_COLUMN].isin(done_ids)].reset_index(drop=True)

print(f"Rows to process: {len(all_rows):,}")
if len(all_rows) == 0:
    dbutils.notebook.exit("All rows already processed.")

# COMMAND ----------

# DBTITLE 1,Async query function with retry and backoff
async def query_one(session, sem, row_id, vec, url, headers, counters):
    payload = {
        "query_vector": vec.tolist() if hasattr(vec, "tolist") else list(vec),
        "columns": [ID_COLUMN],
        "num_results": TOP_K,
    }

    async with sem:
        for attempt in range(MAX_RETRIES):
            started = time.perf_counter()
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    latency_ms = (time.perf_counter() - started) * 1000

                    if resp.status in (429, 503):
                        counters["rate_limited"] += 1
                        await asyncio.sleep(min(0.5 * (2 ** attempt), 30.0))
                        continue

                    if resp.status != 200:
                        counters["errors"] += 1
                        return row_id, [], latency_ms

                    data = await resp.json()
                    result = data.get("result", {})
                    cols = result.get("column_names", [])
                    rows = result.get("data_array", [])
                    score_key = "score" if "score" in cols else (cols[-1] if cols else None)

                    hits = [
                        {
                            "neighbor_id": str(dict(zip(cols, row)).get(ID_COLUMN, "")),
                            "search_score": float(dict(zip(cols, row)).get(score_key, 0.0)) if score_key else 0.0,
                        }
                        for row in rows
                    ]
                    return row_id, hits, latency_ms

            except (aiohttp.ClientError, asyncio.TimeoutError):
                counters["errors"] += 1
                if attempt == MAX_RETRIES - 1:
                    return row_id, [], 0.0
                await asyncio.sleep(0.5 * (2 ** attempt))

    return row_id, [], 0.0

# COMMAND ----------

# DBTITLE 1,Adaptive concurrency controller and main inference loop
current_concurrency = INITIAL_CONCURRENCY

def adjust_concurrency(counters, batch_size):
    global current_concurrency
    rate = counters["rate_limited"] / max(batch_size, 1)
    if rate > BACKOFF_THRESHOLD:
        current_concurrency = max(MIN_CONCURRENCY, current_concurrency // 2)
    elif rate < SCALE_UP_THRESHOLD:
        current_concurrency = int(current_concurrency * 1.25)  # no ceiling


async def run_inference_job():
    global current_concurrency

    url = f"{HOST}/api/2.0/vector-search/indexes/{INDEX_NAME}/query"
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    connector = aiohttp.TCPConnector(limit=0)  # no ceiling — VS rate-limiting drives backoff

    total_rows = len(all_rows)
    processed = 0
    total_latency_ms = 0.0
    total_requests = 0
    started = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=30)) as session:
        for batch_id, start in enumerate(range(0, total_rows, BATCH_SIZE)):
            batch = all_rows.iloc[start : start + BATCH_SIZE]
            counters = {"rate_limited": 0, "errors": 0}
            sem = asyncio.Semaphore(current_concurrency)

            results = await asyncio.gather(*[
                query_one(session, sem, row_id, vec, url, headers, counters)
                for row_id, vec in zip(batch[ID_COLUMN].tolist(), batch[VECTOR_COLUMN].tolist())
            ])

            batch_latencies = [latency for _, _, latency in results if latency > 0]
            total_latency_ms += sum(batch_latencies)
            total_requests += len(batch_latencies)

            # One row per query — neighbors collected into arrays
            rows = []
            processed_at = datetime.now(timezone.utc)
            for row_id, hits, _ in results:
                rows.append({
                    ID_COLUMN:             row_id,
                    "inference_month":     INFERENCE_MONTH,
                    "matched_account_ids": [h["neighbor_id"] for h in hits],
                    "search_scores":       [h["search_score"] for h in hits],
                    "processed_at":        processed_at,
                })

            if rows:
                spark.createDataFrame(pd.DataFrame(rows), schema=OUTPUT_SCHEMA).write.format("delta").mode("append").saveAsTable(OUTPUT_TABLE)

            processed += len(batch)
            elapsed_s = max(time.perf_counter() - started, 1e-9)
            avg_latency = (sum(batch_latencies) / len(batch_latencies)) if batch_latencies else 0.0
            qps = processed / elapsed_s
            pct = processed / total_rows * 100

            print(
                f"[batch {batch_id + 1}] {processed:,}/{total_rows:,} ({pct:.1f}%) | "
                f"avg={avg_latency:.1f}ms | qps={qps:.1f} | concurrency={current_concurrency} | "
                f"rate_limited={counters['rate_limited']} | errors={counters['errors']}"
            )

            adjust_concurrency(counters, len(batch))

    elapsed_s = time.perf_counter() - started
    overall_avg = total_latency_ms / total_requests if total_requests else 0.0
    print(f"Completed {processed:,} queries in {elapsed_s:.1f}s | avg={overall_avg:.1f}ms | qps={processed / elapsed_s:.1f}")
    print(f"Results written to: {OUTPUT_TABLE}")

await run_inference_job()

# COMMAND ----------

# DBTITLE 1,Output table summary
display(spark.sql(f"""
SELECT
    COUNT(*)                              AS total_queries,
    AVG(SIZE(matched_account_ids))        AS avg_matches_per_query,
    AVG(search_scores[0])                 AS avg_top1_score,
    MIN(processed_at)                     AS started_at,
    MAX(processed_at)                     AS finished_at
FROM {OUTPUT_TABLE}
"""))
