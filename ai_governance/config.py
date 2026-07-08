# Local config for the AI Governance demo (values copied from claims_demo/config.py
# so this folder is self-contained when synced as a bundle).

CATALOG = "shm_skunkworks_catalog"
SCHEMA = "claims_demo"
CLAIMS_TABLE = f"{CATALOG}.{SCHEMA}.claims"
NOTES_TABLE = f"{CATALOG}.{SCHEMA}.claim_notes"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.doc_chunks"
VS_INDEX = f"{CATALOG}.{SCHEMA}.doc_chunks_index"

CHAT_MODEL = "databricks-claude-sonnet-4-5"
EMBED_MODEL = "databricks-gte-large-en"
VS_ENDPOINT = "vs"

# Governance-demo objects
GATEWAY_ENDPOINT = "claims-llm-gateway"       # our own endpoint, gateway attached
GATEWAY_INFERENCE_SCHEMA = SCHEMA             # inference tables land in claims_demo
LOOKUP_FUNCTION = f"{CATALOG}.{SCHEMA}.claim_lookup"   # UC function -> managed MCP tool
MASK_FUNCTION = f"{CATALOG}.{SCHEMA}.mask_ssn"         # UC function -> managed MCP tool
