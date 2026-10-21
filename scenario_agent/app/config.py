# Config for the scenario-simulation agent demo.
# Single source of truth, kept local so the folder is self-contained and so
# `mlflow.pyfunc.log_model(code_paths=["config.py"])` packages it with the agent.

CATALOG = "shm_skunkworks_catalog"
SCHEMA = "scenario_demo"

# Foundation Model API endpoints. Discover live names with
#   databricks serving-endpoints list | jq ...
# these are sensible defaults (highest Claude Sonnet for chat, GTE-large for embeddings).
CHAT_MODEL = "databricks-claude-sonnet-4-5"
EMBED_MODEL = "databricks-gte-large-en"

# Delta tables (built by 00_setup.py from seeded synthetic data).
RUNS_TABLE = f"{CATALOG}.{SCHEMA}.simulation_runs"      # run metadata / status
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.simulation_output"  # metric response per scenario/step
SPECS_TABLE = f"{CATALOG}.{SCHEMA}.model_specs"         # model spec text (discoverable)
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.methodology_chunks"  # VS source for doc RAG

# Vector Search (managed embeddings via Delta-Sync).
VS_ENDPOINT = "scenario_vs"
VS_INDEX = f"{CATALOG}.{SCHEMA}.methodology_chunks_index"

# UC SQL functions, exposed to the agent as governed tools via the managed
# Functions MCP server (/api/2.0/mcp/functions/{catalog}/{schema}).
READ_RUNS_FN = f"{CATALOG}.{SCHEMA}.read_runs"
READ_SIM_FN = f"{CATALOG}.{SCHEMA}.read_sim"
DISCOVER_MODELS_FN = f"{CATALOG}.{SCHEMA}.discover_models"
UC_TOOL_FUNCTIONS = [READ_RUNS_FN, READ_SIM_FN, DISCOVER_MODELS_FN]

# Registered UC model + serving endpoint for the deployed agent.
UC_MODEL = f"{CATALOG}.{SCHEMA}.scenario_agent"
ENDPOINT = "scenario-agent"
