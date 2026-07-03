# Databricks notebook source
# MAGIC %md
# MAGIC # VS Bench -- 03: Databricks Vector Search vs Snowflake Cortex Search
# MAGIC
# MAGIC A **transparent, apples-to-apples latency benchmark** between Databricks
# MAGIC Vector Search and Snowflake Cortex Search, driven entirely from Databricks.
# MAGIC
# MAGIC **What it measures.** End-to-end *text query -> top-k results* round-trip
# MAGIC latency. Both services embed the query internally, so we send the same raw
# MAGIC query string to each and time the full call. Same query set, same `top_k`,
# MAGIC same warmup, one call per operation -- no hidden batching, no caching tricks.
# MAGIC
# MAGIC **Two things you can do here:**
# MAGIC 1. **(a)** Benchmark a Databricks Vector Search index against *any existing
# MAGIC    table* -- just point `BENCH_INDEX` at it -- or fall back to the bundled
# MAGIC    ~1M-row `dbpedia` benchmark dataset built by `2a`/`2b`.
# MAGIC 2. **(b)** Call a Snowflake Cortex Search endpoint *from Databricks* and time
# MAGIC    it. A live service is **required** — fill in `SF_*` in `config.py` and
# MAGIC    store a `pat` secret. With no connection, the Snowflake portion fails
# MAGIC    loudly rather than reporting a fabricated number.
# MAGIC
# MAGIC **Portable by design.** The exact REST body and `SEARCH_PREVIEW` SQL we send
# MAGIC to Snowflake are printed at the bottom so a Snowflake engineer can paste them
# MAGIC into a Snowflake worksheet and reproduce the same numbers natively.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   databricks-vectorsearch>=0.44.0 databricks-sdk>=0.49.0 \
# MAGIC   snowflake-connector-python>=3.12.0 requests>=2.32.0
# MAGIC %restart_python

# COMMAND ----------

import datetime as dt
import json
import os
import sys
from time import perf_counter

import pandas as pd
import requests
from databricks.sdk import WorkspaceClient

sys.path.insert(0, os.getcwd())
import config as C

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters
# MAGIC
# MAGIC Widgets override `config.py` so you can re-point the benchmark without
# MAGIC editing code. Leave `bench_index` blank to use the dbpedia 1M index.

# COMMAND ----------

dbutils.widgets.text("bench_index", C.BENCH_INDEX or "", "Databricks VS index (blank = dbpedia 1M)")
dbutils.widgets.text("text_column", C.BENCH_TEXT_COLUMN, "Source text column")
dbutils.widgets.text("id_column", C.BENCH_ID_COLUMN, "Primary key column")
dbutils.widgets.text("top_k", str(C.BENCH_TOPK), "top_k")
dbutils.widgets.text("n_queries", str(C.BENCH_QUERIES), "Benchmark queries")
dbutils.widgets.text("warmup", str(C.BENCH_WARMUP), "Warmup queries")

BENCH_INDEX = dbutils.widgets.get("bench_index").strip() or C.vs_index_name("1m", "ppt")
TEXT_COLUMN = dbutils.widgets.get("text_column").strip()
ID_COLUMN = dbutils.widgets.get("id_column").strip()
TOP_K = int(dbutils.widgets.get("top_k"))
N_QUERIES = int(dbutils.widgets.get("n_queries"))
WARMUP = int(dbutils.widgets.get("warmup"))

print(f"Databricks index : {BENCH_INDEX}")
print(f"top_k={TOP_K}  queries={N_QUERIES}  warmup={WARMUP}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query set
# MAGIC
# MAGIC Held-out query strings -- the same list is sent to *both* backends. We pull
# MAGIC from a query table if configured, else from the dbpedia source (ids the
# MAGIC index never saw, `id >= 900000`).

# COMMAND ----------

if C.BENCH_QUERY_TABLE:
    queries = (spark.table(C.BENCH_QUERY_TABLE)
               .select(TEXT_COLUMN).limit(N_QUERIES)
               .toPandas()[TEXT_COLUMN].astype(str).tolist())
else:
    queries = (spark.table(f"{C.CATALOG}.{C.SCHEMA}.dbpedia_source")
               .where("id >= 900000").orderBy("id").limit(N_QUERIES)
               .select("text").toPandas()["text"].astype(str).tolist())

# Trim very long strings so we benchmark retrieval, not payload size.
queries = [q[:C.TEXT_MAX_CHARS] for q in queries if q]
print(f"Loaded {len(queries)} queries. Example: {queries[0][:120]!r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Backend 1 -- Databricks Vector Search
# MAGIC
# MAGIC We send the raw query **text** and let the index's managed embedding model
# MAGIC handle embedding, so the call is directly comparable to Cortex Search. If
# MAGIC your index is self-managed (precomputed vectors only), set `text_column`
# MAGIC to a column that has a managed embedding, or adapt to pass `query_vector`.

# COMMAND ----------

def search_databricks(text):
    """One end-to-end text->results query against Databricks Vector Search."""
    t0 = perf_counter()
    resp = w.vector_search_indexes.query_index(
        index_name=BENCH_INDEX,
        query_text=text,
        columns=[ID_COLUMN],
        num_results=TOP_K,
    )
    ms = (perf_counter() - t0) * 1000
    n = len((resp.result.data_array if resp.result else None) or [])
    return {"search_ms": ms, "n_results": n}

# Smoke test -- fail loud if the index isn't reachable / not text-queryable.
_probe = search_databricks(queries[0])
print(f"Databricks probe: {_probe['search_ms']:.1f} ms, {_probe['n_results']} hits")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Backend 2 -- Snowflake Cortex Search (called from Databricks)
# MAGIC
# MAGIC Two real code paths:
# MAGIC - **REST**: `POST .../cortex-search-services/{service}:query` -- the path an
# MAGIC   app would use. This is what we time.
# MAGIC - **SQL**: `SNOWFLAKE.CORTEX.SEARCH_PREVIEW(...)` -- printed at the end so it
# MAGIC   ports verbatim into a Snowflake worksheet.
# MAGIC
# MAGIC Auth is read from Databricks secrets (`SF_SECRET_SCOPE`) -- a Programmatic
# MAGIC Access Token is the simplest; keypair-JWT works too (swap the header).
# MAGIC Nothing is hardcoded. **A live service is required**: if the identifiers or
# MAGIC the `pat` secret are missing, this cell raises -- the benchmark never
# MAGIC reports a number it did not actually measure.

# COMMAND ----------

def _cortex_rest_body(text):
    """The exact JSON Cortex Search expects -- identical for REST and SQL."""
    return {"query": text, "columns": [ID_COLUMN], "limit": TOP_K}

# Fail loudly and early if the Snowflake connection isn't fully configured.
_missing = [k for k in ("SF_ACCOUNT", "SF_DATABASE", "SF_SCHEMA", "SF_SERVICE")
            if not getattr(C, k)]
if _missing:
    raise ValueError(
        f"Snowflake not configured -- set {_missing} in config.py and store a "
        f"`pat` secret in scope '{C.SF_SECRET_SCOPE}'. This benchmark only times "
        f"a real Cortex Search service; there is no mock/simulated mode."
    )

SF_PAT = dbutils.secrets.get(C.SF_SECRET_SCOPE, "pat")  # programmatic access token
SF_BASE = f"https://{C.SF_ACCOUNT}.snowflakecomputing.com"
SF_URL = (f"{SF_BASE}/api/v2/databases/{C.SF_DATABASE}/schemas/{C.SF_SCHEMA}"
          f"/cortex-search-services/{C.SF_SERVICE}:query")
_sf_session = requests.Session()
_sf_session.headers.update({
    "Authorization": f"Bearer {SF_PAT}",
    "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    "Content-Type": "application/json",
    "Accept": "application/json",
})

def search_snowflake(text):
    body = _cortex_rest_body(text)
    t0 = perf_counter()
    r = _sf_session.post(SF_URL, data=json.dumps(body), timeout=30)
    ms = (perf_counter() - t0) * 1000
    r.raise_for_status()
    rows = r.json().get("results", [])
    return {"search_ms": ms, "n_results": len(rows)}

_sf_probe = search_snowflake(queries[0])
print(f"Snowflake probe: {_sf_probe['search_ms']:.1f} ms, {_sf_probe['n_results']} hits")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run: warmup + benchmark
# MAGIC
# MAGIC Identical loop for every backend -- the only thing that differs is the
# MAGIC `search_*` function, which keeps the comparison honest.

# COMMAND ----------

BACKENDS = {"databricks_vs": search_databricks, "snowflake_cortex": search_snowflake}

rows = []
for backend, search in BACKENDS.items():
    print(f"  {backend}: warmup ({WARMUP}) + bench ({len(queries)})")
    for q in queries[:WARMUP]:
        search(q)
    for i, q in enumerate(queries):
        r = search(q)
        rows.append({
            "backend": backend, "query_idx": i,
            "search_ms": r["search_ms"], "n_results": r.get("n_results"),
        })

results = pd.DataFrame(rows)
print(f"{len(results):,} measurements collected")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate -- p50 / p90 / p99

# COMMAND ----------

def q(p):
    return lambda s: s.quantile(p)

agg = (results.groupby("backend")
       .agg(n=("search_ms", "size"),
            p50_ms=("search_ms", q(0.50)),
            p90_ms=("search_ms", q(0.90)),
            p99_ms=("search_ms", q(0.99)),
            mean_ms=("search_ms", "mean"),
            min_ms=("search_ms", "min"),
            max_ms=("search_ms", "max"))
       .reset_index().round(1))
print(agg.to_string(index=False))
try:
    display(agg)  # noqa: F821
except NameError:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist results

# COMMAND ----------

stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
out_tbl = f"{C.CATALOG}.{C.SCHEMA}.cortex_vs_bench_{stamp}"
spark.createDataFrame(results).write.mode("overwrite").saveAsTable(out_tbl)
print(f"Per-query results -> {out_tbl}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Portability -- reproduce the Snowflake side natively
# MAGIC
# MAGIC Paste either of these into a Snowflake worksheet (SQL) or any HTTP client
# MAGIC (REST). They send the *same* query body this notebook uses, so timings line
# MAGIC up. In Snowflake, wrap the SQL with the query profile / `QUERY_HISTORY` to
# MAGIC read server-side latency.

# COMMAND ----------

_example = queries[0].replace("'", "''")
_body = _cortex_rest_body(queries[0])

print("=== Snowflake native SQL (SEARCH_PREVIEW) ===\n")
print(f"""SELECT PARSE_JSON(
  SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    '{C.SF_DATABASE or "<db>"}.{C.SF_SCHEMA or "<schema>"}.{C.SF_SERVICE or "<service>"}',
    '{json.dumps(_body)}'
  )
)['results'] AS results;""")

print("\n\n=== Snowflake REST (curl) ===\n")
print(f"""curl -X POST \\
  "https://<account>.snowflakecomputing.com/api/v2/databases/{C.SF_DATABASE or "<db>"}\\
/schemas/{C.SF_SCHEMA or "<schema>"}/cortex-search-services/{C.SF_SERVICE or "<service>"}:query" \\
  -H "Authorization: Bearer <PAT>" \\
  -H "X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(_body)}'""")

