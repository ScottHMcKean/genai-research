# Vector Search benchmark config.
# Plain Python so notebooks, scripts, and local tools all import the same thing.
# No YAML, no notebook-path magic.

CATALOG = "shm"
SCHEMA = "genai"
VOLUME = "hf_cache"

# Source dataset (only used by 2a_setup_data.py)
HF_DATASET = "KShivendu/dbpedia-entities-openai-1M"
TEXT_MAX_CHARS = 2048

# Embedding models -- both 1024-dim
EMBED_DIM = 1024
PPT_MODEL = "databricks-gte-large-en"
PT_HF_MODEL = "Qwen/Qwen3-Embedding-0.6B"
PT_UC_MODEL = "shm.genai.qwen3_emb_06b"
PT_ENDPOINT = "shm_qwen3_emb_06b"

# Scales. Flip DEV_MODE to run the full benchmark.
DEV_MODE = False
DEV_SCALES = ["10k"]
SCALES = {"10k": 10_000, "100k": 100_000, "1m": 1_000_000}
MODELS = ["ppt", "pt"]

# Per-scale model overrides. Scales not listed here use `MODELS`.
# `1m/pt` is skipped because our single-replica GPU_SMALL Qwen endpoint
# can't embed 1M rows within reasonable task timeouts.
SCALE_MODELS = {
    "1m": ["ppt"],
}

# Databricks Vector Search
VS_ENDPOINT = "one-env-shared-endpoint-1"

# Lakebase
LAKEBASE_PROJECT = "vs-bench"
LAKEBASE_BRANCH = "production"
LAKEBASE_ENDPOINT = "primary"
LAKEBASE_DB = "vsbench"       # default `postgres` db can't host CREATE EXTENSION
# Profile is only used when running the Databricks CLI locally. Inside a
# Databricks notebook there's no ~/.databrickscfg -- the CLI auto-authenticates
# via DATABRICKS_HOST / OAuth env vars, so we omit -p PROFILE entirely.
LAKEBASE_PROFILE = "DEFAULT"

# Benchmark
BENCH_QUERIES = 200
BENCH_WARMUP = 20
BENCH_TOPK = 10
HNSW_EF_SEARCH = 80

# ---------------------------------------------------------------------------
# Cortex-vs-VectorSearch benchmark (notebook 3)
# ---------------------------------------------------------------------------
# Databricks side: by default we benchmark a managed-embedding index so the
# comparison is end-to-end text->results, exactly like Cortex Search (which
# embeds the query internally). Point BENCH_INDEX at any existing index, or
# leave it None to fall back to the dbpedia 1M index from 2b_setup_vs.py.
BENCH_INDEX = None                 # e.g. "shm.genai.dbpedia_1m_ppt_vsidx"
BENCH_TEXT_COLUMN = "text"         # source text column on the index
BENCH_ID_COLUMN = "id"             # primary key column returned by queries
BENCH_QUERY_TABLE = None           # table to pull held-out query strings from;
                                   # None -> dbpedia_source (id >= 900000)

# Snowflake Cortex Search side.
# Auth is read from Databricks secrets at run time -- nothing is hardcoded.
# A live Cortex Search service is REQUIRED: the benchmark times real calls only.
# Fill these in and store a `pat` (programmatic access token) in the secret
# scope before running the Snowflake portion.
SF_SECRET_SCOPE = "snowflake"      # dbutils.secrets scope holding SF creds
SF_ACCOUNT = ""                    # e.g. "ab12345.us-east-1"
SF_DATABASE = ""                   # database holding the Cortex Search service
SF_SCHEMA = ""                     # schema holding the service
SF_SERVICE = ""                    # Cortex Search service name


def active_scales():
    """Return the scales dict to iterate over based on DEV_MODE."""
    if DEV_MODE:
        return {k: SCALES[k] for k in DEV_SCALES if k in SCALES}
    return dict(SCALES)


def models_for(scale):
    return SCALE_MODELS.get(scale, MODELS)


def combos():
    """All (scale, model) pairs active for this run."""
    return [(s, m) for s in active_scales() for m in models_for(s)]


def embed_table(scale, model):
    return f"{CATALOG}.{SCHEMA}.dbpedia_{scale}_{model}"


def vs_index_name(scale, model):
    return f"{CATALOG}.{SCHEMA}.dbpedia_{scale}_{model}_vsidx"


def pg_table_name(scale, model):
    return f"dbpedia_{scale}_{model}"
