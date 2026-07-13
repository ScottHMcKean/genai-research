# Databricks notebook source

# MAGIC %md
# MAGIC # Vector Search: Filter Queries with Managed Embeddings
# MAGIC
# MAGIC **Solution:** When the index is configured with **managed embeddings**
# MAGIC (Databricks computes embeddings via a model serving endpoint), the
# MAGIC `query_text` parameter works natively -- the platform embeds the query
# MAGIC text and performs similarity search.
# MAGIC
# MAGIC This notebook demonstrates the managed embedding approach and compares
# MAGIC query behavior with self-managed embeddings.
# MAGIC
# MAGIC | Index type | `query_text` | `query_vector` | Filters | Best for |
# MAGIC |---|---|---|---|---|
# MAGIC | Delta Sync (managed) | Works | Works | Pre-filter (standard) | Most use cases |
# MAGIC | Delta Sync (self-managed) | Fails (no endpoint) | Works | Pre-filter (standard) | Custom embeddings |
# MAGIC | Direct Access | Fails (no endpoint) | Works | Pre-filter (standard) | Real-time upserts |
# MAGIC
# MAGIC **See also:** `1a_self_managed_filter_test` for the self-managed embedding
# MAGIC problem reproduction.

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
# MAGIC
# MAGIC `databricks-gte-large-en` is a Foundation Model API endpoint available on
# MAGIC all Databricks workspaces. It produces 1024-dimensional embeddings.

# COMMAND ----------

ENDPOINT_NAME = "one-env-shared-endpoint-1"
EMBEDDING_MODEL = "databricks-gte-large-en"
EMBEDDING_DIM = 1024
CATALOG = "shm"
SCHEMA = "default"
TABLE_NAME = f"{CATALOG}.{SCHEMA}.vs_filter_test_people_text"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.vs_filter_test_managed_idx"
NUM_PEOPLE = 50

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create test data (text only -- no pre-computed embeddings)
# MAGIC
# MAGIC With managed embeddings, you only need a text column. Databricks
# MAGIC computes the embeddings automatically during index sync.

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
    jt = job_titles[i % len(job_titles)]
    rows.append({
        "id": i,
        "first_name": fn,
        "last_name": ln,
        "email": f"{fn.lower()}{i}@example.net",
        "job_title": jt,
        "bio": f"{fn} {ln} is a {jt} with 10 years of experience in data and analytics.",
    })

target = rows[4]
print(f"Target: id={target['id']} email={target['email']} "
      f"name={target['first_name']} {target['last_name']} "
      f"job={target['job_title']}")

# COMMAND ----------

from pyspark.sql import types as T

spark_schema = T.StructType([
    T.StructField("id", T.IntegerType()),
    T.StructField("first_name", T.StringType()),
    T.StructField("last_name", T.StringType()),
    T.StructField("email", T.StringType()),
    T.StructField("job_title", T.StringType()),
    T.StructField("bio", T.StringType()),
])

df = spark.createDataFrame(rows, schema=spark_schema)

spark.sql(f"DROP TABLE IF EXISTS {TABLE_NAME}")
(df.write
   .format("delta")
   .option("delta.enableChangeDataFeed", "true")
   .saveAsTable(TABLE_NAME))

spark.sql(f"ALTER TABLE {TABLE_NAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"Created table {TABLE_NAME} with {df.count()} rows (CDF enabled)")
display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Delta Sync index with managed embeddings
# MAGIC
# MAGIC The key difference from self-managed: we specify `embedding_source_columns`
# MAGIC with a model endpoint, instead of `embedding_vector_columns` with
# MAGIC pre-computed vectors. Databricks computes and stores the embeddings.

# COMMAND ----------

from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
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

print(f"Creating managed embedding index: {INDEX_NAME}")
print(f"  source_table: {TABLE_NAME}")
print(f"  embedding_model: {EMBEDDING_MODEL}")
print(f"  text_column: bio")

w.vector_search_indexes.create_index(
    name=INDEX_NAME,
    endpoint_name=ENDPOINT_NAME,
    primary_key="id",
    index_type=VectorIndexType.DELTA_SYNC,
    delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
        source_table=TABLE_NAME,
        embedding_source_columns=[
            EmbeddingSourceColumn(
                name="bio",
                embedding_model_endpoint_name=EMBEDDING_MODEL,
            )
        ],
        pipeline_type=PipelineType.TRIGGERED,
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for index to sync
# MAGIC
# MAGIC The sync pipeline reads the Delta table, calls the embedding model for
# MAGIC each row, and stores the resulting vectors. This takes a few minutes.

# COMMAND ----------

deadline = time.time() + 900
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
    raise TimeoutError("Index did not sync within 15 minutes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run filter tests
# MAGIC
# MAGIC With managed embeddings, `query_text` works because the platform has an
# MAGIC endpoint to embed it.

# COMMAND ----------

results_log = []

def timed_query(label, **kwargs):
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
# MAGIC ### TEST 1: Semantic query + filter (the natural way)
# MAGIC
# MAGIC With managed embeddings, this is the intended usage: pass natural language
# MAGIC in `query_text`, optionally add filters. The platform embeds the query and
# MAGIC finds similar records that also match the filter.

# COMMAND ----------

print(f"TEST 1: query_text='analyst' + email filter")
result = timed_query(
    "semantic query + filter",
    query_text="analyst",
    columns=["id", "first_name", "last_name", "email", "job_title", "bio"],
    filters_json=json.dumps({"email": target["email"]}),
    num_results=4,
)
for row in result["rows"]:
    print(f"  id={row[0]}  email={row[3]}  job={row[4]}  score={row[-1]}  "
          f"latency={result['latency_ms']:.0f}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ### TEST 2: `query_text=""` + filter (the customer's original attempt)
# MAGIC
# MAGIC With managed embeddings, this technically works -- the empty string gets
# MAGIC embedded. The score is the similarity to the empty-string embedding, which
# MAGIC is semantically meaningless but the filter still returns the right rows.

# COMMAND ----------

print("TEST 2: query_text='' + email filter (managed embeddings)")
try:
    result = timed_query(
        "query_text='' + filter (managed)",
        query_text="",
        columns=["id", "first_name", "last_name", "email", "job_title"],
        filters_json=json.dumps({"email": target["email"]}),
        num_results=4,
    )
    for row in result["rows"]:
        print(f"  id={row[0]}  email={row[3]}  score={row[-1]}  "
              f"latency={result['latency_ms']:.0f}ms")
    print("query_text='' WORKS with managed embeddings (score is noise)")
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### TEST 3: Semantic query without filter (pure similarity search)

# COMMAND ----------

print("TEST 3: query_text='data engineer' without filter")
result = timed_query(
    "semantic query, no filter",
    query_text="data engineer",
    columns=["id", "first_name", "last_name", "email", "job_title"],
    num_results=5,
)
for row in result["rows"]:
    print(f"  id={row[0]}  name={row[1]} {row[2]}  job={row[4]}  score={row[-1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### TEST 4: Semantic query + non-unique filter (job_title)

# COMMAND ----------

print(f"TEST 4: query_text='experienced professional' + job_title filter")
result = timed_query(
    "semantic + job_title filter",
    query_text="experienced professional",
    columns=["id", "first_name", "last_name", "email", "job_title"],
    filters_json=json.dumps({"job_title": "Data Scientist"}),
    num_results=10,
)
print(f"Returned {len(result['rows'])} Data Scientists:")
for row in result["rows"]:
    print(f"  id={row[0]}  name={row[1]} {row[2]}  job={row[4]}  score={row[-1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### TEST 5: Hybrid search + filter

# COMMAND ----------

print("TEST 5: hybrid search (semantic + keyword) + email filter")
try:
    result = timed_query(
        "hybrid + filter",
        query_text="analyst data",
        query_type="hybrid",
        columns=["id", "first_name", "last_name", "email", "job_title"],
        filters_json=json.dumps({"email": target["email"]}),
        num_results=4,
    )
    for row in result["rows"]:
        print(f"  id={row[0]}  email={row[3]}  job={row[4]}  score={row[-1]}")
except Exception as e:
    print(f"Hybrid search error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### TEST 6: SQL on the source Delta table (exact-match baseline)

# COMMAND ----------

t0 = time.perf_counter()
sql_result = spark.sql(f"""
    SELECT id, first_name, last_name, email, job_title
    FROM {TABLE_NAME}
    WHERE email = '{target["email"]}'
""")
sql_rows = sql_result.collect()
sql_ms = (time.perf_counter() - t0) * 1000
print(f"SQL returned {len(sql_rows)} rows in {sql_ms:.0f}ms")
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
# MAGIC These mirror the customer's original `curl`-based approach using raw
# MAGIC HTTP requests against the Vector Search REST API.

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
# MAGIC ### REST TEST 1: `query_text` + filter (managed -- should work)
# MAGIC
# MAGIC With managed embeddings, the platform embeds the query text server-side.

# COMMAND ----------

print("REST TEST 1: query_text='analyst' + email filter")
resp, ms = timed_api_call("query_text + filter", {
    "columns": ["id", "first_name", "last_name", "email", "job_title"],
    "num_results": 4,
    "filters_json": json.dumps({"email": target["email"]}),
    "query_text": "analyst",
})
if resp.ok:
    for row in resp.json()["result"]["data_array"]:
        print(f"  id={row[0]}  email={row[3]}  score={row[-1]}  latency={ms:.0f}ms")
else:
    print(f"ERROR: {resp.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### REST TEST 2: `query_text=""` + filter (the customer's original curl)

# COMMAND ----------

print("REST TEST 2: query_text='' + email filter")
resp, ms = timed_api_call("query_text='' + filter", {
    "columns": ["id", "first_name", "last_name", "email", "job_title"],
    "num_results": 4,
    "filters_json": json.dumps({"email": target["email"]}),
    "query_text": "",
})
if resp.ok:
    for row in resp.json()["result"]["data_array"]:
        print(f"  id={row[0]}  email={row[3]}  score={row[-1]}  latency={ms:.0f}ms")
    print("query_text='' WORKS with managed embeddings via REST API")
else:
    print(f"ERROR: {resp.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### REST TEST 3: hybrid search + filter

# COMMAND ----------

print("REST TEST 3: hybrid search + filter via REST")
resp, ms = timed_api_call("hybrid + filter", {
    "columns": ["id", "first_name", "last_name", "email", "job_title"],
    "num_results": 4,
    "filters_json": json.dumps({"email": target["email"]}),
    "query_text": "analyst data",
    "query_type": "hybrid",
})
if resp.ok:
    for row in resp.json()["result"]["data_array"]:
        print(f"  id={row[0]}  email={row[3]}  score={row[-1]}  latency={ms:.0f}ms")
else:
    print(f"ERROR: {resp.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### REST TEST 4: filter only via REST (no query_text, no query_vector)

# COMMAND ----------

print("REST TEST 4: filter only, no query_text or query_vector")
resp, ms = timed_api_call("filter only (no query)", {
    "columns": ["id", "first_name", "last_name", "email", "job_title"],
    "num_results": 4,
    "filters_json": json.dumps({"email": target["email"]}),
})
if resp.ok:
    for row in resp.json()["result"]["data_array"]:
        print(f"  id={row[0]}  email={row[3]}  score={row[-1]}  latency={ms:.0f}ms")
else:
    print(f"ERROR ({resp.status_code}): {resp.text[:200]}")

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
# MAGIC ## Recommendations
# MAGIC
# MAGIC ### Which approach should the customer use?
# MAGIC
# MAGIC | Use case | Recommendation |
# MAGIC |---|---|
# MAGIC | **Exact-match lookup** (find by email, ID) | SQL on the Delta table -- fastest, simplest |
# MAGIC | **Semantic search** (find similar bios) | Managed embeddings with `query_text` |
# MAGIC | **Semantic search + filtering** | Managed embeddings + `filters_json` |
# MAGIC | **Must keep Azure OpenAI embeddings** | Pass `query_vector` from Azure OpenAI; use dummy vector for pure-filter only |
# MAGIC
# MAGIC ### Endpoint type considerations
# MAGIC
# MAGIC | | Standard | Storage-Optimized |
# MAGIC |---|---|---|
# MAGIC | Latency | ~50-100ms | ~250ms |
# MAGIC | Filter behavior | **Pre-filter** (guaranteed match) | **Post-filter** (not guaranteed) |
# MAGIC | Capacity | 320M vectors (768d) | 1B+ vectors (768d) |
# MAGIC | Cost | Higher | ~7x lower |
# MAGIC | Best for | Real-time apps, exact filters | Large-scale, cost-sensitive |
# MAGIC
# MAGIC **Critical for the customer:** On storage-optimized endpoints, the dummy
# MAGIC vector workaround is **not reliable** -- the filter is applied after
# MAGIC fetching top-k results by similarity, so a record with low similarity to
# MAGIC the dummy vector may not be in the top-k set even though it matches the
# MAGIC filter.
# MAGIC
# MAGIC ### Migration path from self-managed to managed embeddings
# MAGIC
# MAGIC 1. Remove the pre-computed embedding column from the source table
# MAGIC 2. Add a text column (or reuse existing) that contains the content to embed
# MAGIC 3. Create a new Delta Sync index with `embedding_source_columns` pointing to
# MAGIC    a Databricks model endpoint (`databricks-gte-large-en` for English text)
# MAGIC 4. `query_text` now works natively -- no more dummy vectors
