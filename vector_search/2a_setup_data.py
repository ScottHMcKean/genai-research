# Databricks notebook source
# MAGIC %md
# MAGIC # VS Bench -- 01: Data + Embeddings (idempotent)
# MAGIC
# MAGIC Runs only what's missing:
# MAGIC 1. Source Delta table from `KShivendu/dbpedia-entities-openai-1M`
# MAGIC 2. Scale-subset tables
# MAGIC 3. Qwen3-Embedding-0.6B pyfunc + GPU serving endpoint
# MAGIC 4. Embedding tables (`dbpedia_{scale}_{model}`) via `ai_query`
# MAGIC
# MAGIC If an embedding table already has the expected row count and dim, it is left alone.
# MAGIC Safe to re-run — skips finished work.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   huggingface_hub>=0.26.0 pyarrow>=19.0.0 \
# MAGIC   sentence-transformers>=3.0.0 transformers>=4.45.0 torch \
# MAGIC   mlflow>=2.22.0 databricks-sdk>=0.49.0
# MAGIC %restart_python

# COMMAND ----------

import glob, os, sys, time
import pandas as pd
import pyarrow.parquet as pq
import mlflow
from huggingface_hub import snapshot_download
from databricks.sdk import WorkspaceClient
from mlflow.deployments import get_deploy_client
from mlflow.models import infer_signature

sys.path.insert(0, os.getcwd())
import config as C

VOL = f"/Volumes/{C.CATALOG}/{C.SCHEMA}/{C.VOLUME}"
SRC = f"{C.CATALOG}.{C.SCHEMA}.dbpedia_source"

w = WorkspaceClient()
dc = get_deploy_client("databricks")
print(f"scales={list(C.active_scales())} models={C.MODELS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Source Delta (1M rows) -- skip if already populated

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {C.CATALOG}.{C.SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {C.CATALOG}.{C.SCHEMA}.{C.VOLUME}")

if spark.catalog.tableExists(SRC) and spark.table(SRC).count() == 1_000_000:
    print(f"{SRC} already has 1M rows -- skipping HF download")
else:
    ds_dir = f"{VOL}/dbpedia_raw"
    snapshot_download(C.HF_DATASET, repo_type="dataset", local_dir=ds_dir,
                      allow_patterns=["*.parquet"])
    files = sorted(glob.glob(f"{ds_dir}/**/*.parquet", recursive=True))
    spark.sql(f"DROP TABLE IF EXISTS {SRC}")
    total = 0
    for p in files:
        t = pq.read_table(p, columns=["title", "text"])
        titles, texts = t.column("title").to_pylist(), t.column("text").to_pylist()
        rows = [{
            "id": total + i,
            "title": (titles[i] or "").strip()[:512],
            "text": ((titles[i] or "") + " -- " + (texts[i] or "")).strip(" -")[:C.TEXT_MAX_CHARS],
        } for i in range(len(titles))]
        (spark.createDataFrame(rows)
             .write.mode("append").format("delta")
             .option("delta.enableChangeDataFeed", "true")
             .saveAsTable(SRC))
        total += len(rows)
        print(f"  ingested {total:,}/1,000,000")
    assert spark.table(SRC).count() == 1_000_000

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Scale-subset tables

# COMMAND ----------

for scale, n in C.active_scales().items():
    tbl = f"{C.CATALOG}.{C.SCHEMA}.dbpedia_{scale}_src"
    if spark.catalog.tableExists(tbl) and spark.table(tbl).count() == n:
        print(f"  {tbl}: already {n:,} rows -- skip")
        continue
    spark.sql(f"""
        CREATE OR REPLACE TABLE {tbl}
        TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
        SELECT id, title, text FROM {SRC} ORDER BY id LIMIT {n}
    """)
    print(f"  {tbl}: {n:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Qwen custom embedding endpoint -- skip if endpoint is already READY

# COMMAND ----------

def endpoint_ready(name):
    try:
        ep = w.serving_endpoints.get(name)
        return ep.state and ep.state.ready and ep.state.ready.value == "READY"
    except Exception:
        return False

if endpoint_ready(C.PT_ENDPOINT):
    print(f"  endpoint {C.PT_ENDPOINT} already READY -- skip deploy")
else:
    qwen_dir = f"{VOL}/qwen3-emb-06b"
    if not os.path.exists(f"{qwen_dir}/config.json"):
        snapshot_download(C.PT_HF_MODEL, local_dir=qwen_dir)

    class QwenEmbedder(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(context.artifacts["repo"], device=device)

        def predict(self, context, model_input, params=None):
            if isinstance(model_input, pd.DataFrame):
                texts = model_input.iloc[:, 0].astype(str).tolist()
            else:
                texts = [str(x) for x in (model_input if isinstance(model_input, list) else [model_input])]
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    mlflow.set_registry_uri("databricks-uc")
    with mlflow.start_run(run_name="qwen3_emb_06b"):
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=QwenEmbedder(),
            artifacts={"repo": qwen_dir},
            pip_requirements=["torch==2.4.0", "sentence-transformers>=3.0.0",
                              "transformers>=4.45.0", "accelerate>=0.30.0"],
            input_example=pd.DataFrame({"text": ["hello world"]}),
            signature=infer_signature(
                pd.DataFrame({"text": ["hello world"]}),
                [[0.0] * C.EMBED_DIM],
            ),
            registered_model_name=C.PT_UC_MODEL,
        )
    version = info.registered_model_version

    entities = [{"name": C.PT_ENDPOINT, "entity_name": C.PT_UC_MODEL,
                 "entity_version": version, "workload_type": "GPU_SMALL",
                 "workload_size": "Small", "scale_to_zero_enabled": False}]
    try:
        dc.update_endpoint(endpoint=C.PT_ENDPOINT, config={"served_entities": entities})
    except Exception:
        dc.create_endpoint(name=C.PT_ENDPOINT, config={"served_entities": entities})

    deadline = time.time() + 1800
    while time.time() < deadline:
        if endpoint_ready(C.PT_ENDPOINT):
            break
        print("  waiting for endpoint ready...")
        time.sleep(30)
    else:
        raise TimeoutError("Qwen endpoint not READY after 30 min")
    print(f"  endpoint {C.PT_ENDPOINT} ready @ v{version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Embedding tables -- skip any that already have the expected rows + dim

# COMMAND ----------

def embed_table_ok(tbl, expected_n):
    if not spark.catalog.tableExists(tbl):
        return False
    try:
        n = spark.table(tbl).count()
        dim = spark.sql(f"SELECT size(embedding) FROM {tbl} LIMIT 1").collect()[0][0]
        return n == expected_n and dim == C.EMBED_DIM
    except Exception:
        return False

for scale, n in C.active_scales().items():
    for model in C.models_for(scale):
        tgt = C.embed_table(scale, model)
        if embed_table_ok(tgt, n):
            print(f"  {tgt}: {n:,} rows, dim={C.EMBED_DIM} -- skip")
            continue
        ep = C.PPT_MODEL if model == "ppt" else C.PT_ENDPOINT
        src = f"{C.CATALOG}.{C.SCHEMA}.dbpedia_{scale}_src"
        t0 = time.perf_counter()
        spark.sql(f"""
            CREATE OR REPLACE TABLE {tgt}
            TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
            SELECT id, title, text,
                   CAST(ai_query('{ep}', text, returnType => 'ARRAY<FLOAT>') AS ARRAY<FLOAT>) AS embedding
            FROM {src}
        """)
        dim = spark.sql(f"SELECT size(embedding) FROM {tgt} LIMIT 1").collect()[0][0]
        assert dim == C.EMBED_DIM
        print(f"  {tgt}: {n:,} rows, dim={dim} in {time.perf_counter()-t0:.1f}s")
