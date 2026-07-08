# Shared config for the insurance-and-claims demo suite.
# Plain Python so every notebook imports the same names.
#
# Three demos build on this one claims spine:
#   - agent_bricks_claims/  (KA + Genie + Supervisor + custom RAG)
#   - ai_governance/        (AI Gateway + guardrails + MCP + inference logging)
#   - ai_runtime/           (Ray on serverless GPU + LoRA fine-tune for triage)

CATALOG = "shm_skunkworks_catalog"
SCHEMA = "claims_demo"
VOLUME = "docs"                       # knowledge docs (policy wordings, guidelines) for the KA/RAG
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# Tables
CLAIMS_TABLE = f"{CATALOG}.{SCHEMA}.claims"           # structured claims -> Genie, PII guardrails, FT labels
NOTES_TABLE = f"{CATALOG}.{SCHEMA}.claim_notes"       # free-text adjuster notes -> RAG / FT text
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.doc_chunks"       # chunked knowledge docs -> Vector Search source
VS_INDEX = f"{CATALOG}.{SCHEMA}.doc_chunks_index"     # Vector Search index (Delta Sync)

# Foundation models (serverless, pay-per-token -- confirmed available on the FEVM)
CHAT_MODEL = "databricks-claude-sonnet-4-5"
EMBED_MODEL = "databricks-gte-large-en"
EMBED_DIM = 1024

# Vector Search endpoint -- dedicated endpoint for the claims demo (avoids sharing
# compute with the large pre-existing index on the shared `vs` endpoint)
VS_ENDPOINT = "claims_vs"

# Scale (kept small on purpose -- "simple but showcases the platform")
N_CLAIMS = 2000
SEED = 42
