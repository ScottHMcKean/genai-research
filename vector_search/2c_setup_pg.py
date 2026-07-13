# Databricks notebook source
# MAGIC %md
# MAGIC # VS Bench -- 03: Lakebase pgvector indices
# MAGIC
# MAGIC Uses the Databricks Python SDK (`api_client.do`) against the autoscaling
# MAGIC postgres REST API. The `databricks` CLI is explicitly blocked on serverless
# MAGIC compute, so we never shell out.
# MAGIC
# MAGIC Per-table idempotent: if `dbpedia_{scale}_{model}` already has the expected
# MAGIC rows and an HNSW index, it is left alone.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk>=0.49.0 "psycopg[binary,pool]>=3.2.0"
# MAGIC %restart_python

# COMMAND ----------

import os, sys, time
from io import StringIO
import pandas as pd
import psycopg
from databricks.sdk import WorkspaceClient

sys.path.insert(0, os.getcwd())
import config as C

w = WorkspaceClient()
api = w.api_client
print(f"scales={list(C.active_scales())} models={C.MODELS}")

PROJECT_PATH = f"projects/{C.LAKEBASE_PROJECT}"
BRANCH_PATH = f"{PROJECT_PATH}/branches/{C.LAKEBASE_BRANCH}"
ENDPOINT_PATH = f"{BRANCH_PATH}/endpoints/{C.LAKEBASE_ENDPOINT}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure project exists (create + wait for LRO if missing)

# COMMAND ----------

def get_project():
    try:
        return api.do("GET", f"/api/2.0/postgres/{PROJECT_PATH}")
    except Exception as e:
        if type(e).__name__ == "NotFound" or "not found" in str(e).lower() or "404" in str(e):
            return None
        raise

def create_project():
    op = api.do(
        "POST", "/api/2.0/postgres/projects",
        query={"project_id": C.LAKEBASE_PROJECT},
        body={"spec": {"display_name": "VS Benchmark"}},
    )
    op_name = op.get("name", "")
    if not op_name:
        return
    deadline = time.time() + 300
    while time.time() < deadline:
        status = api.do("GET", f"/api/2.0/postgres/{op_name}")
        if status.get("done"):
            if status.get("error"):
                raise RuntimeError(f"create-project failed: {status['error']}")
            return
        time.sleep(5)
    raise TimeoutError("create-project LRO did not complete in 5 min")

if get_project() is None:
    print(f"Creating project {C.LAKEBASE_PROJECT}...")
    create_project()
print(f"project {C.LAKEBASE_PROJECT} ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure primary endpoint ACTIVE; scale to 2-16 CU

# COMMAND ----------

def list_endpoints():
    resp = api.do("GET", f"/api/2.0/postgres/{BRANCH_PATH}/endpoints")
    return resp.get("endpoints", []) if isinstance(resp, dict) else (resp or [])

def endpoint_info(eps):
    if not eps:
        return None, None, None, None
    s = eps[0].get("status", {}) or {}
    spec = eps[0].get("spec", {}) or {}
    return (s.get("current_state") or s.get("state"),
            (s.get("hosts") or {}).get("host"),
            spec.get("autoscaling_limit_min_cu"),
            spec.get("autoscaling_limit_max_cu"))

def wait_ready(dwell_checks=3, per_sleep=10, timeout=900):
    """Wait until endpoint state is ACTIVE or IDLE (host available, can accept
    connections). Autoscaling endpoints settle into IDLE when suspended; the
    first incoming connection wakes them up."""
    READY = {"ACTIVE", "IDLE"}
    deadline = time.time() + timeout
    streak = 0
    last_host = None
    while time.time() < deadline:
        state, host, _, _ = endpoint_info(list_endpoints())
        if state in READY and host:
            last_host = host
            streak += 1
            if streak >= dwell_checks:
                return host
        else:
            streak = 0
            print(f"  endpoint state: {state}")
        time.sleep(per_sleep)
    raise TimeoutError(f"Endpoint not ready (last={last_host})")

host = wait_ready()
print(f"endpoint host: {host}")

# status.autoscaling_limit_{min,max}_cu reflects the current applied range;
# spec mirrors intent. Read from status since spec can be null on freshly
# autoscaled endpoints.
s, _, _, _ = endpoint_info(list_endpoints())
eps = list_endpoints()
status = eps[0].get("status", {}) if eps else {}
cur_min = status.get("autoscaling_limit_min_cu")
cur_max = status.get("autoscaling_limit_max_cu")
if cur_min != 2 or cur_max != 16:
    print(f"  patching CU range {cur_min}-{cur_max} -> 2-16")
    api.do(
        "PATCH", f"/api/2.0/postgres/{ENDPOINT_PATH}",
        query={"update_mask": "spec.autoscaling_limit_min_cu,spec.autoscaling_limit_max_cu"},
        body={"spec": {"autoscaling_limit_min_cu": 2.0, "autoscaling_limit_max_cu": 16.0}},
    )
    time.sleep(15)
    host = wait_ready()
    print(f"  post-patch ready @ {host}")
else:
    print(f"  CU range already {cur_min}-{cur_max}, skipping patch")

# COMMAND ----------

# MAGIC %md
# MAGIC ## pg connection helper (OAuth token via REST)

# COMMAND ----------

email = w.current_user.me().user_name

def pg_conn(database=C.LAKEBASE_DB, retries=3):
    """Connect with retry on transient disconnects (e.g. endpoint scaling)."""
    last = None
    for i in range(retries):
        try:
            tok = api.do("POST", "/api/2.0/postgres/credentials",
                         body={"endpoint": ENDPOINT_PATH})["token"]
            return psycopg.connect(host=host, port=5432, dbname=database, user=email,
                                   password=tok, sslmode="require", autocommit=True)
        except psycopg.OperationalError as e:
            last = e
            print(f"  pg_conn retry {i+1}/{retries}: {e}")
            time.sleep(5 * (i + 1))
    raise last

def run_pg_sql(database, sql, params=None, retries=3):
    """Run a single statement with retry on AdminShutdown/disconnect."""
    last = None
    for i in range(retries):
        try:
            with pg_conn(database) as conn, conn.cursor() as cur:
                cur.execute(sql, params) if params else cur.execute(sql)
                try:
                    return cur.fetchall()
                except psycopg.ProgrammingError:
                    return None
        except (psycopg.OperationalError, psycopg.errors.AdminShutdown) as e:
            last = e
            print(f"  run_pg_sql retry {i+1}/{retries}: {type(e).__name__}: {e}")
            time.sleep(10 * (i + 1))
    raise last

# Default `postgres` db is too restricted (no CREATE EXTENSION privilege),
# so we create our own. autocommit=True means CREATE DATABASE runs cleanly
# outside any transaction.
exists = run_pg_sql("postgres",
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (C.LAKEBASE_DB,))
if not exists:
    run_pg_sql("postgres", f"CREATE DATABASE {C.LAKEBASE_DB}")
run_pg_sql(C.LAKEBASE_DB, "CREATE EXTENSION IF NOT EXISTS vector")
print(f"pgvector ready in {C.LAKEBASE_DB}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load tables + build HNSW (skip tables already loaded)

# COMMAND ----------

CHUNK = 50_000

def pg_table_ready(cur, tbl, expected_n):
    cur.execute("SELECT to_regclass(%s)", (tbl,))
    if cur.fetchone()[0] is None:
        return False
    cur.execute(f"SELECT count(*) FROM {tbl}")
    if cur.fetchone()[0] != expected_n:
        return False
    cur.execute(
        "SELECT 1 FROM pg_indexes WHERE tablename = %s AND indexname = %s",
        (tbl, f"{tbl}_hnsw"),
    )
    return cur.fetchone() is not None

def pgvec(arr):
    return "[" + ",".join(f"{float(x):.6f}" for x in arr) + "]"

with pg_conn() as conn, conn.cursor() as cur:
    cur.execute("SET synchronous_commit = OFF")
    for scale, model in C.combos():
        tbl = C.pg_table_name(scale, model)
        n = C.active_scales()[scale]
        if pg_table_ready(cur, tbl, n):
            print(f"  {tbl}: already loaded ({n:,} rows + HNSW) -- skip")
            continue

        src = C.embed_table(scale, model)
        cur.execute(f"DROP TABLE IF EXISTS {tbl}")
        cur.execute(f"""
            CREATE TABLE {tbl} (
                id BIGINT PRIMARY KEY, title TEXT, text TEXT,
                embedding vector({C.EMBED_DIM})
            )
        """)

        t0 = time.perf_counter()
        n_chunks = max(1, n // CHUNK)
        for c in range(n_chunks):
            pdf = (spark.table(src).where(f"pmod(id, {n_chunks}) = {c}")
                   .select("id", "title", "text", "embedding").toPandas())
            # psycopg's typed COPY handles all text escaping for us;
            # vector is sent as its text literal "[...]" which postgres casts.
            with cur.copy(f"COPY {tbl} (id,title,text,embedding) FROM STDIN") as cp:
                for _, row in pdf.iterrows():
                    cp.write_row((int(row["id"]),
                                  row["title"] or "",
                                  row["text"] or "",
                                  pgvec(row["embedding"])))
        load_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        cur.execute(f"""CREATE INDEX {tbl}_hnsw ON {tbl}
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m=16, ef_construction=64)""")
        cur.execute(f"ANALYZE {tbl}")
        print(f"  {tbl}: load={load_s:.1f}s, hnsw={time.perf_counter()-t0:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Memory sanity check (HNSW vs shared_buffers)

# COMMAND ----------

with pg_conn() as conn, conn.cursor() as cur:
    cur.execute("SELECT setting::bigint * 8192 FROM pg_settings WHERE name='shared_buffers'")
    sb_bytes = cur.fetchone()[0]
    rows = []
    for scale, model in C.combos():
        tbl = C.pg_table_name(scale, model)
        cur.execute(f"""SELECT pg_relation_size('{tbl}'::regclass),
                               pg_relation_size('{tbl}_hnsw'::regclass)""")
        tb, ib = cur.fetchone()
        rows.append({"scale": scale, "model": model,
                     "table_mb": round(tb/1024/1024, 1),
                     "index_mb": round(ib/1024/1024, 1)})

df = pd.DataFrame(rows)
total_idx_mb = df["index_mb"].sum()
sb_mb = round(sb_bytes/1024/1024, 1)
df["shared_buffers_mb"] = sb_mb
print(df.to_string(index=False))
if total_idx_mb > sb_mb:
    print(f"WARNING: HNSW indices ({total_idx_mb:.0f} MB) > shared_buffers ({sb_mb:.0f} MB) -- queries may spill")
else:
    print(f"OK: HNSW fits in shared_buffers ({total_idx_mb:.0f}/{sb_mb:.0f} MB)")
