# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Vector Search index over the claims knowledge docs
# MAGIC
# MAGIC Builds a **Delta-Sync** Vector Search index on `doc_chunks` using **managed
# MAGIC embeddings** (`databricks-gte-large-en`). This is the retriever behind the custom
# MAGIC RAG agent. Reuses the existing `vs` endpoint on the FEVM.
# MAGIC
# MAGIC Showcases: serverless Vector Search, managed embeddings, Unity Catalog governance,
# MAGIC automatic sync from a Delta table (CDF).

# COMMAND ----------

# MAGIC %pip install --quiet databricks-vectorsearch
# MAGIC %restart_python

# COMMAND ----------

from config import CHUNKS_TABLE, VS_INDEX, VS_ENDPOINT, EMBED_MODEL
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

existing = [i.get("name") for i in vsc.list_indexes(name=VS_ENDPOINT).get("vector_indexes", [])]
if VS_INDEX not in existing:
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX,
        source_table_name=CHUNKS_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="chunk",
        embedding_model_endpoint_name=EMBED_MODEL,
    )
    print("Creating index (builds asynchronously):", VS_INDEX)
else:
    print("Index already exists:", VS_INDEX)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait until the index is ONLINE, then trigger a sync

# COMMAND ----------

import time

idx = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
for _ in range(40):
    status = idx.describe().get("status", {})
    state = status.get("detailed_state", status.get("state", ""))
    ready = status.get("ready", False)
    print("state:", state, "| ready:", ready)
    if ready:
        break
    time.sleep(30)

# Keep the index fresh with the latest chunks
try:
    idx.sync()
    print("Sync triggered.")
except Exception as e:
    print("Sync note:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test — a claims question

# COMMAND ----------

res = idx.similarity_search(
    query_text="What is the procedure for a basement water damage claim?",
    columns=["doc", "title", "chunk"],
    num_results=3,
)
for row in res.get("result", {}).get("data_array", []):
    print("•", row[0], "—", row[1])
    print("  ", row[2][:160].replace("\n", " "), "...\n")
