# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup — synthetic data, Vector Search index, UC tools
# MAGIC
# MAGIC Builds everything the agent needs, from seeded synthetic data (no real data, no
# MAGIC external fetches):
# MAGIC 1. `simulation_runs`, `simulation_output`, `model_specs` Delta tables (CDF on).
# MAGIC 2. `methodology_chunks` + a Delta-Sync **Vector Search** index (managed embeddings).
# MAGIC 3. The `read_runs`, `read_sim`, `discover_models` **UC SQL functions** (the agent's
# MAGIC    governed tools, exposed via the managed Functions MCP server).
# MAGIC
# MAGIC Run on serverless. Edit `config.py` to point at a catalog/schema you can write to.
# MAGIC
# MAGIC No `%pip install` / `restartPython` here: everything uses packages already present on
# MAGIC serverless (numpy, pyspark, the Databricks SDK). Installing and restarting the kernel
# MAGIC to pull `databricks-vectorsearch` downgrades protobuf and breaks Spark Connect, so we
# MAGIC create the Vector Search index with the **SDK** instead.

# COMMAND ----------

import uuid
from pathlib import Path

import numpy as np
from pyspark.sql import functions as F

import config

CAT, SCH = config.CATALOG, config.SCHEMA
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CAT}.{SCH}")
print(f"Target schema: {CAT}.{SCH}")

rng = np.random.default_rng(42)  # seeded → reproducible demo data

# COMMAND ----------

# MAGIC %md ## 1. Synthetic simulation data
# MAGIC Three models, four runs. Each run exercises a scenario set against a handful of
# MAGIC metrics over 21 steps; metric responses are impulse-response-like decaying curves.

# COMMAND ----------

MODELS = [
    ("growth_a", "2026.1", "Baseline growth model with monetary and demand channels.", 3),
    ("growth_b", "2026.1", "Growth model variant with a steeper demand response.", 3),
    ("stress_v", "2026.1", "Stress model used for adverse-scenario runs.", 3),
]
SCENARIOS = ["demand_up", "rate_up", "supply_down"]
METRICS = ["output", "inflation", "rate"]
STEPS = list(range(21))

RUNS = [
    ("run_0001", "growth_a", "standard", "complete", "2026-10-01T09:00:00Z", "2026-10-01T09:04:00Z", "monthly refresh"),
    ("run_0002", "growth_b", "standard", "complete", "2026-10-03T14:10:00Z", "2026-10-03T14:15:00Z", "demand-response variant"),
    ("run_0003", "stress_v", "adverse",  "complete", "2026-10-07T08:30:00Z", "2026-10-07T08:38:00Z", "adverse stress run"),
    ("run_0004", "growth_a", "standard", "running",  "2026-10-18T16:00:00Z", None,                   "re-run in progress"),
]


def _curve(amp, tau, osc):
    """Seeded impulse-response-like path over STEPS: a decaying, mildly oscillating pulse."""
    s = np.array(STEPS, dtype=float)
    base = amp * np.exp(-s / tau) * np.cos(osc * s)
    return base + rng.normal(0, 0.02 * abs(amp) + 1e-6, size=s.shape)


runs_df = spark.createDataFrame(
    RUNS, schema=["run_id", "model", "scenario_set", "status", "submitted_at", "completed_at", "note"]
)

output_rows = []
for run_id, model, *_ in RUNS:
    if run_id == "run_0004":
        continue  # still running → no output yet (exercises the agent's "nothing returned" path)
    for sc in SCENARIOS:
        for mt in METRICS:
            amp = {"output": 1.0, "inflation": 0.5, "rate": 0.25}[mt] * (1.3 if model == "growth_b" else 1.0)
            if sc == "supply_down":
                amp *= -1
            vals = _curve(amp, tau=6.0, osc=0.25)
            for step, v in zip(STEPS, vals):
                output_rows.append((run_id, sc, mt, step, round(float(v), 5)))

output_df = spark.createDataFrame(
    output_rows, schema=["run_id", "scenario", "metric", "step", "value"]
)

specs_df = spark.createDataFrame(
    [(m, v, d, n) for (m, v, d, n) in MODELS],
    schema=["model_name", "version", "description", "num_scenarios"],
)


def _write(df, table):
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"  wrote {table} ({df.count()} rows)")


_write(runs_df, config.RUNS_TABLE)
_write(output_df, config.OUTPUT_TABLE)
_write(specs_df, config.SPECS_TABLE)

# COMMAND ----------

# MAGIC %md ## 2. Methodology docs → chunks → Vector Search index

# COMMAND ----------

DOCS = {
    "overview.md": (
        "# Scenario simulation overview\n\n"
        "A simulation run applies a scenario set to a model and records how each metric "
        "responds over a sequence of steps. Steps are relative horizons, not calendar "
        "dates. Metrics include output, inflation, and rate. A run has a status "
        "(running or complete) and belongs to exactly one model."
    ),
    "scenarios.md": (
        "# Scenario definitions\n\n"
        "demand_up is a positive demand impulse. rate_up is a policy-rate impulse. "
        "supply_down is an adverse supply impulse and typically moves output negative. "
        "Scenario names are stable identifiers; always read exact names from read_sim "
        "discovery mode rather than guessing."
    ),
    "reading_results.md": (
        "# Reading results\n\n"
        "Use discover_models to list models, read_runs to find runs and their status, and "
        "read_sim to read metric responses. Call read_sim first without metrics to see "
        "which scenarios and metrics exist, then again with exact metrics for the numbers. "
        "Values are aligned to a shared steps array."
    ),
}


def _chunk(text, size=800, overlap=120):
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


chunk_rows = []
for src, text in DOCS.items():
    for i, ch in enumerate(_chunk(text)):
        chunk_rows.append((str(uuid.uuid4()), ch, src, i))

chunks_df = spark.createDataFrame(
    chunk_rows, schema=["chunk_id", "chunk", "source_uri", "chunk_position"]
).withColumn("indexed_at", F.current_timestamp())
_write(chunks_df, config.CHUNKS_TABLE)

# COMMAND ----------

# Create the endpoint + Delta-Sync index via the Databricks SDK (no extra pip install).
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    EndpointType,
    PipelineType,
    VectorIndexType,
)

w = WorkspaceClient()

try:
    w.vector_search_endpoints.get_endpoint(config.VS_ENDPOINT)
    print(f"VS endpoint '{config.VS_ENDPOINT}' exists.")
except Exception:
    print(f"Creating VS endpoint '{config.VS_ENDPOINT}' (provisioning may take minutes)...")
    w.vector_search_endpoints.create_endpoint_and_wait(
        name=config.VS_ENDPOINT, endpoint_type=EndpointType.STANDARD)

try:
    w.vector_search_indexes.get_index(config.VS_INDEX)
    print("Index exists; triggering sync.")
    w.vector_search_indexes.sync_index(config.VS_INDEX)
except Exception:
    print(f"Creating Delta-Sync index '{config.VS_INDEX}'...")
    w.vector_search_indexes.create_index(
        name=config.VS_INDEX,
        endpoint_name=config.VS_ENDPOINT,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=config.CHUNKS_TABLE,
            pipeline_type=PipelineType.TRIGGERED,
            embedding_source_columns=[EmbeddingSourceColumn(
                name="chunk", embedding_model_endpoint_name=config.EMBED_MODEL)],
        ),
    )
    print("Index created.")

# COMMAND ----------

# MAGIC %md ## 3. Register the UC SQL function tools

# COMMAND ----------


def _sql_dir() -> Path:
    for c in [Path.cwd() / "sql", Path.cwd() / "scenario_agent" / "sql"]:
        if c.exists():
            return c
    try:
        d = Path(__file__).resolve().parent / "sql"  # noqa: F821 (defined in files/notebooks)
        if d.exists():
            return d
    except NameError:
        pass
    raise FileNotFoundError("Could not locate sql/ — run this notebook from the scenario_agent folder.")


SQL_DIR = _sql_dir()
for name in ("read_runs", "read_sim", "discover_models"):
    ddl = (SQL_DIR / f"{name}.sql").read_text()
    ddl = ddl.replace("{{catalog}}", CAT).replace("{{schema}}", SCH)
    spark.sql(ddl)
    print(f"  registered {CAT}.{SCH}.{name}")

# COMMAND ----------

# MAGIC %md ## 4. Smoke-test the tools

# COMMAND ----------

print(spark.sql(f"SELECT {CAT}.{SCH}.discover_models() AS out").first()["out"][:400])
print(spark.sql(f"SELECT {CAT}.{SCH}.read_runs(NULL, 10) AS out").first()["out"][:400])
print(spark.sql(
    f"SELECT {CAT}.{SCH}.read_sim(array('run_0001'), NULL, NULL, NULL, NULL, 2000) AS out"
).first()["out"][:500])
print("\nSetup complete.")
