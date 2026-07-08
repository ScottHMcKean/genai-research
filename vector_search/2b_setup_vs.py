# Databricks notebook source
# MAGIC %md
# MAGIC # VS Bench -- 02: Databricks Vector Search indices
# MAGIC
# MAGIC Builds a Standard Delta Sync index on `VS_ENDPOINT` for each `(scale, model)`.
# MAGIC Idempotent: keeps existing indices as-is if they're already reporting ready
# MAGIC with the expected row count.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-vectorsearch>=0.44.0 databricks-sdk>=0.49.0
# MAGIC %restart_python

# COMMAND ----------

import os, sys, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest, EmbeddingVectorColumn,
    PipelineType, VectorIndexType,
)

sys.path.insert(0, os.getcwd())
import config as C

w = WorkspaceClient()
print(f"scales={list(C.active_scales())} models={C.MODELS} endpoint={C.VS_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create missing indices (skip ones already ready at full row count)

# COMMAND ----------

def index_get(name):
    try:
        return w.vector_search_indexes.get_index(index_name=name)
    except Exception:
        return None

def index_ready_rows(idx):
    s = idx.status if idx else None
    return (bool(s and s.ready), int(s.indexed_row_count or 0) if s else 0)

def wait_gone(name, timeout=600):
    """Poll until get_index returns nothing (or raises NotFound)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if index_get(name) is None:
            return
        print(f"  {name}: waiting for deletion to complete...")
        time.sleep(10)
    raise TimeoutError(f"{name} still exists after deletion timeout")

def create_index(name, src):
    w.vector_search_indexes.create_index(
        name=name, endpoint_name=C.VS_ENDPOINT, primary_key="id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=src, pipeline_type=PipelineType.TRIGGERED,
            embedding_vector_columns=[
                EmbeddingVectorColumn(name="embedding", embedding_dimension=C.EMBED_DIM),
            ],
        ),
    )

to_wait = []
for scale, model in C.combos():
    name = C.vs_index_name(scale, model)
    src = C.embed_table(scale, model)
    expected = C.active_scales()[scale]

    idx = index_get(name)
    if idx:
        ready, rows = index_ready_rows(idx)
        if ready and rows >= expected:
            print(f"  {name}: ready ({rows:,} rows) -- skip")
            continue
        # Exists but not usable -- delete and wait for delete to truly complete
        # before create, otherwise create races with async deletion.
        print(f"  {name}: exists but not ready -- deleting")
        try:
            w.vector_search_indexes.delete_index(index_name=name)
        except Exception as e:
            print(f"    delete said: {e}")
        wait_gone(name)

    create_index(name, src)
    to_wait.append(name)
    print(f"  {name}: creating")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for readiness

# COMMAND ----------

deadline = time.time() + 1800
while to_wait and time.time() < deadline:
    still = []
    for name in to_wait:
        ready, rows = index_ready_rows(index_get(name))
        if ready and rows > 0:
            print(f"  {name}: ready ({rows:,} rows)")
        else:
            still.append(name)
    to_wait = still
    if to_wait:
        time.sleep(20)
if to_wait:
    raise TimeoutError(f"VS indices not ready: {to_wait}")
print("All VS indices ready.")
