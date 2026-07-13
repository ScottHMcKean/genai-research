# Databricks notebook source

# MAGIC %md
# MAGIC # Vector Search: Filter Queries with Self-Managed Embeddings
# MAGIC
# MAGIC **Problem:** A customer has a Delta Sync index with self-managed embeddings
# MAGIC (pre-computed via Azure OpenAI, 1536 dimensions). They want to do a pure
# MAGIC filter-only query (e.g., find a record by email) without a semantic search
# MAGIC component. When they pass `query_text: ""`, the API returns:
# MAGIC
# MAGIC > `query vector must be specified for index ... as it is a direct access
# MAGIC > index without a model serving endpoint or self-managed delta sync index.`
# MAGIC
# MAGIC This notebook reproduces the issue and tests workarounds:
# MAGIC
# MAGIC | Test | Approach | Works? |
# MAGIC |------|----------|--------|
# MAGIC | 1 | `query_text=""` + filter | No -- no endpoint to embed empty string |
# MAGIC | 2 | Dummy `query_vector` + filter | Yes -- filter is pre-applied, score is noise |
# MAGIC | 3 | Dummy vector without filter | Yes -- shows top-k by dummy similarity |
# MAGIC | 4 | Real embedding + filter | Yes -- score = 1.0 (exact match) |
# MAGIC | 5 | Zero vector + filter | Yes -- score is noise |
# MAGIC | 6 | `scan_index` (no vector) | Yes -- but no server-side filtering |
# MAGIC | 7 | SQL on source Delta table | Yes -- best option for exact-match lookups |
# MAGIC
# MAGIC **See also:** `1b_managed_embedding_filter_test` for the managed embedding
# MAGIC approach where `query_text` works natively.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import random
import time

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

ENDPOINT_NAME = "one-env-shared-endpoint-1"
CATALOG = "shm"
SCHEMA = "default"
TABLE_NAME = f"{CATALOG}.{SCHEMA}.vs_filter_test_people"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.vs_filter_test_self_managed_idx"
EMBEDDING_DIM = 128
NUM_PEOPLE = 50

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create test data with pre-computed embeddings
# MAGIC
# MAGIC Simulates the customer's scenario: a Delta table that already has an
# MAGIC embedding column computed by an external model (Azure OpenAI in their case).

# COMMAND ----------

random.seed(42)

first_names = ["Allison", "Brandon", "Cynthia", "David", "Elijah",
               "Fiona", "George", "Hannah", "Ivan", "Julia",
               "Kevin", "Laura", "Michael", "Nancy", "Oscar"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones",
              "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
              "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson"]
job_titles = ["Software Engineer", "Data Scientist", "Product Manager",
              "Analyst", "DevOps Engineer", "Accountant", "Marketing Manager",
              "Solutions Architect", "VP Engineering", "Data Engineer",
              "ML Engineer", "QA Engineer", "Technical Writer", "CEO", "CFO"]

rows = []
for i in range(NUM_PEOPLE):
    fn = first_names[i % len(first_names)]
    ln = last_names[i % len(last_names)]
    rows.append({
        "id": i,
        "first_name": fn,
        "last_name": ln,
        "email": f"{fn.lower()}{i}@example.net",
        "job_title": job_titles[i % len(job_titles)],
        "bio": f"{fn} {ln} works as a {job_titles[i % len(job_titles)]}.",
        "embedding": [random.gauss(0, 1) for _ in range(EMBEDDING_DIM)],
    })

target = rows[4]
print(f"Target for tests: id={target['id']} email={target['email']} "
      f"name={target['first_name']} {target['last_name']}")

# COMMAND ----------

from pyspark.sql import types as T

spark_schema = T.StructType([
    T.StructField("id", T.IntegerType()),
    T.StructField("first_name", T.StringType()),
    T.StructField("last_name", T.StringType()),
    T.StructField("email", T.StringType()),
    T.StructField("job_title", T.StringType()),
    T.StructField("bio", T.StringType()),
    T.StructField("embedding", T.ArrayType(T.FloatType())),
])

df = spark.createDataFrame(rows, schema=spark_schema)

spark.sql(f"DROP TABLE IF EXISTS {TABLE_NAME}")
(df.write
   .format("delta")
   .option("delta.enableChangeDataFeed", "true")
   .saveAsTable(TABLE_NAME))

spark.sql(f"ALTER TABLE {TABLE_NAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"Created table {TABLE_NAME} with {df.count()} rows (CDF enabled)")
display(df.drop("embedding").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Delta Sync index with self-managed embeddings
# MAGIC
# MAGIC This mirrors the customer's setup: the embedding column already exists in
# MAGIC the table, and the index does **not** have a model serving endpoint.

# COMMAND ----------

from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    PipelineType,
    VectorIndexType,
)

try:
    w.vector_search_indexes.get_index(index_name=INDEX_NAME)
    print(f"Deleting existing index {INDEX_NAME}")
    w.vector_search_indexes.delete_index(index_name=INDEX_NAME)
    time.sleep(10)
except Exception:
    pass

print(f"Creating delta sync index: {INDEX_NAME}")
w.vector_search_indexes.create_index(
    name=INDEX_NAME,
    endpoint_name=ENDPOINT_NAME,
    primary_key="id",
    index_type=VectorIndexType.DELTA_SYNC,
    delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
        source_table=TABLE_NAME,
        embedding_vector_columns=[
            EmbeddingVectorColumn(
                name="embedding",
                embedding_dimension=EMBEDDING_DIM,
            )
        ],
        pipeline_type=PipelineType.TRIGGERED,
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for index to sync

# COMMAND ----------

deadline = time.time() + 600
while time.time() < deadline:
    idx = w.vector_search_indexes.get_index(index_name=INDEX_NAME)
    ready = idx.status.ready if idx.status else False
    msg = idx.status.message if idx.status else "unknown"
    print(f"  ready={ready}  message={msg}")
    if ready:
        print("Index is ready")
        break
    time.sleep(15)
else:
    raise TimeoutError("Index did not sync within 10 minutes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run filter tests
# MAGIC
# MAGIC Each test measures wall-clock latency and documents the behavior.

# COMMAND ----------

results_log = []

def timed_query(label, **kwargs):
    """Run a query, measure latency, and log the result."""
    t0 = time.perf_counter()
    try:
        resp = w.vector_search_indexes.query_index(index_name=INDEX_NAME, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        rows = resp.result.data_array if resp.result else []
        cols = [c.name for c in resp.manifest.columns] if resp.manifest else []
        results_log.append({
            "test": label, "status": "OK",
            "latency_ms": round(elapsed_ms, 1), "rows": len(rows),
        })
        return {"columns": cols, "rows": rows, "latency_ms": elapsed_ms}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results_log.append({
            "test": label, "status": "ERROR",
            "latency_ms": round(elapsed_ms, 1), "error": str(e)[:120],
        })
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ### TEST 1: `query_text=""` + filter (reproduces the customer error)

# COMMAND ----------

print("TEST 1: query_text='' + email filter")
try:
    result = timed_query(
        "query_text='' + filter",
        query_text="",
        columns=["id", "first_name", "last_name", "email", "job_title"],
        filters_json=json.dumps({"email": target["email"]}),
        num_results=4,
    )
    print(f"Unexpected success: {result}")
except Exception as e:
    print(f"EXPECTED ERROR: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### TEST 2: Dummy `query_vector` + filter (customer's workaround)
# MAGIC
# MAGIC The filter is applied as a **pre-filter** on standard endpoints, so the
# MAGIC matching record is **guaranteed** to be returned. The score column reflects
# MAGIC cosine similarity to the dummy vector, which is meaningless for a
# MAGIC pure-filter use case.

# COMMAND ----------

dummy_vector = [0.5] * EMBEDDING_DIM

print(f"TEST 2: dummy query_vector + filter (email={target['email']})")
result = timed_query(
    "dummy_vector + filter",
    query_vector=dummy_vector,
    columns=["id", "first_name", "last_name", "email", "job_title"],
    filters_json=json.dumps({"email": target["email"]}),
    num_results=4,
)
for row in result["rows"]:
    print(f"  id={row[0]}  email={row[3]}  score={row[-1]}  latency={result['latency_ms']:.0f}ms")
found = any(row[3] == target["email"] for row in result["rows"])
print(f"Target found: {found}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### TEST 3: Dummy vector without filter (baseline ranking)

# COMMAND ----------

print("TEST 3: dummy query_vector, no filter (top-4 by dummy similarity)")
result = timed_query(
    "dummy_vector, no filter",
    query_vector=dummy_vector,
    columns=["id", "first_name", "last_name", "email"],
    num_results=4,
)
for row in result["rows"]:
    print(f"  id={row[0]}  email={row[3]}  score={row[-1]}")
target_in_top4 = any(row[3] == target["email"] for row in result["rows"])
print(f"Target in top-4 without filter: {target_in_top4}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### TEST 4: Real embedding + filter (score = 1.0)

# COMMAND ----------

print("TEST 4: target's real embedding + filter")
result = timed_query(
    "real_vector + filter",
    query_vector=target["embedding"],
    columns=["id", "first_name", "last_name", "email", "job_title"],
    filters_json=json.dumps({"email": target["email"]}),
    num_results=4,
)
for row in result["rows"]:
    print(f"  id={row[0]}  email={row[3]}  score={row[-1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### TEST 5: Zero vector + filter

# COMMAND ----------

print("TEST 5: zero vector + filter")
try:
    result = timed_query(
        "zero_vector + filter",
        query_vector=[0.0] * EMBEDDING_DIM,
        columns=["id", "first_name", "last_name", "email", "job_title"],
        filters_json=json.dumps({"email": target["email"]}),
        num_results=4,
    )
    for row in result["rows"]:
        print(f"  id={row[0]}  email={row[3]}  score={row[-1]}")
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### TEST 6: `scan_index` (no vector at all)
# MAGIC
# MAGIC `scan_index` retrieves all rows from the index without any similarity
# MAGIC search. It does not support server-side filtering -- you filter
# MAGIC client-side. Useful for debugging or bulk export.

# COMMAND ----------

print("TEST 6: scan_index")
t0 = time.perf_counter()
scan = w.vector_search_indexes.scan_index(
    index_name=INDEX_NAME,
    num_results=NUM_PEOPLE,
)
scan_ms = (time.perf_counter() - t0) * 1000
scan_dict = scan.as_dict()
data = scan_dict.get("data", [])
print(f"scan_index returned {len(data)} rows in {scan_ms:.0f}ms")
target_found = any(target["email"] in str(r) for r in data)
print(f"Target found (client-side filter): {target_found}")
results_log.append({
    "test": "scan_index", "status": "OK",
    "latency_ms": round(scan_ms, 1), "rows": len(data),
})

# COMMAND ----------

# MAGIC %md
# MAGIC ### TEST 7: Plain SQL on the source Delta table
# MAGIC
# MAGIC For exact-match lookups, querying the Delta table directly is the simplest
# MAGIC and most efficient approach. No vector search needed.

# COMMAND ----------

t0 = time.perf_counter()
sql_result = spark.sql(f"""
    SELECT id, first_name, last_name, email, job_title
    FROM {TABLE_NAME}
    WHERE email = '{target["email"]}'
""")
sql_rows = sql_result.collect()
sql_ms = (time.perf_counter() - t0) * 1000
print(f"SQL query returned {len(sql_rows)} rows in {sql_ms:.0f}ms")
for r in sql_rows:
    print(f"  id={r.id}  email={r.email}  name={r.first_name} {r.last_name}")
results_log.append({
    "test": "SQL on Delta table", "status": "OK",
    "latency_ms": round(sql_ms, 1), "rows": len(sql_rows),
})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Bare REST API calls
# MAGIC
# MAGIC These reproduce the customer's original `curl` approach using `requests`
# MAGIC against the Vector Search REST API directly, without the SDK wrapper.

# COMMAND ----------

import requests

api_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = api_ctx.apiUrl().get()
token = api_ctx.apiToken().get()

API_URL = f"{host}/api/2.0/vector-search/indexes/{INDEX_NAME}/query"
HEADERS = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def timed_api_call(label, payload):
    t0 = time.perf_counter()
    resp = requests.post(API_URL, headers=HEADERS, json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    status = "OK" if resp.ok else "ERROR"
    body = resp.json() if resp.ok else {}
    num_rows = len(body.get("result", {}).get("data_array", []))
    results_log.append({
        "test": f"[REST] {label}",
        "status": status,
        "latency_ms": round(elapsed_ms, 1),
        "rows": num_rows if resp.ok else 0,
        **({"error": resp.text[:120]} if not resp.ok else {}),
    })
    return resp, elapsed_ms

# COMMAND ----------

# MAGIC %md
# MAGIC ### REST TEST 1: `query_text=""` + filter (customer's curl)
# MAGIC
# MAGIC This is the exact payload the customer sent.

# COMMAND ----------

print("REST TEST 1: query_text='' + filter")
resp, ms = timed_api_call("query_text='' + filter", {
    "columns": ["id", "first_name", "last_name", "email", "job_title"],
    "num_results": 4,
    "filters_json": json.dumps({"email": target["email"]}),
    "query_text": "",
})
if resp.ok:
    for row in resp.json()["result"]["data_array"]:
        print(f"  {row}")
else:
    print(f"EXPECTED ERROR ({resp.status_code}): {resp.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### REST TEST 2: dummy `query_vector` + filter

# COMMAND ----------

print("REST TEST 2: dummy query_vector + filter")
resp, ms = timed_api_call("dummy_vector + filter", {
    "columns": ["id", "first_name", "last_name", "email", "job_title"],
    "num_results": 4,
    "filters_json": json.dumps({"email": target["email"]}),
    "query_vector": [0.5] * EMBEDDING_DIM,
})
if resp.ok:
    for row in resp.json()["result"]["data_array"]:
        print(f"  id={row[0]}  email={row[3]}  score={row[-1]}  latency={ms:.0f}ms")
else:
    print(f"ERROR: {resp.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### REST TEST 3: real embedding + filter

# COMMAND ----------

print("REST TEST 3: real embedding + filter")
resp, ms = timed_api_call("real_vector + filter", {
    "columns": ["id", "first_name", "last_name", "email", "job_title"],
    "num_results": 4,
    "filters_json": json.dumps({"email": target["email"]}),
    "query_vector": target["embedding"],
})
if resp.ok:
    for row in resp.json()["result"]["data_array"]:
        print(f"  id={row[0]}  email={row[3]}  score={row[-1]}  latency={ms:.0f}ms")
else:
    print(f"ERROR: {resp.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### REST TEST 4: scan_index via REST (no vector)

# COMMAND ----------

print("REST TEST 4: scan_index via REST")
scan_url = f"{host}/api/2.0/vector-search/indexes/{INDEX_NAME}/scan"
t0 = time.perf_counter()
resp = requests.get(scan_url, headers=HEADERS, params={"num_results": NUM_PEOPLE})
scan_ms = (time.perf_counter() - t0) * 1000
if resp.ok:
    data = resp.json().get("data", [])
    print(f"scan returned {len(data)} rows in {scan_ms:.0f}ms")
    results_log.append({"test": "[REST] scan_index", "status": "OK",
                        "latency_ms": round(scan_ms, 1), "rows": len(data)})
else:
    print(f"ERROR: {resp.text[:200]}")
    results_log.append({"test": "[REST] scan_index", "status": "ERROR",
                        "latency_ms": round(scan_ms, 1), "error": resp.text[:120]})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Results summary

# COMMAND ----------

import pandas as pd

summary_df = pd.DataFrame(results_log)
print(summary_df.to_string(index=False))
display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Cleanup

# COMMAND ----------

try:
    w.vector_search_indexes.delete_index(index_name=INDEX_NAME)
    print(f"Deleted index {INDEX_NAME}")
except Exception as e:
    print(f"Index cleanup: {e}")

spark.sql(f"DROP TABLE IF EXISTS {TABLE_NAME}")
print(f"Dropped table {TABLE_NAME}")

# COMMAND ----------

dbutils.notebook.exit(json.dumps(results_log))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key findings
# MAGIC
# MAGIC | Endpoint type | Filter behavior | Guaranteed match? |
# MAGIC |---|---|---|
# MAGIC | **Standard** | Pre-filter: filter applied before ANN search | Yes |
# MAGIC | **Storage-Optimized** | Post-filter: filter applied after top-k fetch | No -- matching row may not be in top-k |
# MAGIC
# MAGIC **For pure exact-match lookups** (e.g., find by email):
# MAGIC - Use SQL on the Delta table. It is faster and semantically correct.
# MAGIC - The dummy vector workaround functions, but the score is noise.
# MAGIC
# MAGIC **For semantic search + filtering** (e.g., "find data engineers named Elijah"):
# MAGIC - Switch to managed embeddings (see notebook 02). Then `query_text` works.
# MAGIC - Or embed the query client-side (call Azure OpenAI) and pass `query_vector`.
