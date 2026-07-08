# Config for the Agent Bricks + Custom RAG claims demo.
# Mirrors claims_demo/config.py -- kept local so each demo folder is self-contained
# when synced as a bundle.

CATALOG = "shm_skunkworks_catalog"
SCHEMA = "claims_demo"
VOLUME = "docs"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

CLAIMS_TABLE = f"{CATALOG}.{SCHEMA}.claims"
NOTES_TABLE = f"{CATALOG}.{SCHEMA}.claim_notes"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.doc_chunks"
VS_INDEX = f"{CATALOG}.{SCHEMA}.doc_chunks_index"

CHAT_MODEL = "databricks-claude-sonnet-4-5"
EMBED_MODEL = "databricks-gte-large-en"
EMBED_DIM = 1024
VS_ENDPOINT = "claims_vs"   # dedicated endpoint for the claims demo

# Registered UC model + serving endpoint for the custom RAG agent
RAG_UC_MODEL = f"{CATALOG}.{SCHEMA}.claims_rag_agent"
RAG_ENDPOINT = "claims-rag-agent"
