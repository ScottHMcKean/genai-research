# AI Runtime & Fine-Tuning demo config.
# Plain Python so every notebook imports the same names.
# Builds on the shared claims spine created by claims_demo/00_setup_and_data.py.

CATALOG = "shm_skunkworks_catalog"
SCHEMA = "claims_demo"

# Source (shared spine)
NOTES_TABLE = f"{CATALOG}.{SCHEMA}.claim_notes"      # claim_id, claim_type, peril, severity, note
CLAIMS_TABLE = f"{CATALOG}.{SCHEMA}.claims"

# This demo's tables
TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.triage_train"
TEST_TABLE = f"{CATALOG}.{SCHEMA}.triage_test"
RAY_SCORED_TABLE = f"{CATALOG}.{SCHEMA}.triage_ray_scored"   # Ray distributed LLM triage output
EVAL_TABLE = f"{CATALOG}.{SCHEMA}.triage_eval"               # fine-tuned vs zero-shot comparison

# Fine-tuned model (registered in Unity Catalog)
UC_MODEL = f"{CATALOG}.{SCHEMA}.claims_triage_model"
SERVING_ENDPOINT = "claims-triage-ft"                        # optional serving endpoint name

# Foundation model (serverless, pay-per-token) for the Ray + zero-shot baseline
CHAT_MODEL = "databricks-claude-sonnet-4-5"

# Fine-tuning base model -- small on purpose so a LoRA run finishes in minutes.
# DistilBERT sequence classifier: real fine-tuning, ~66M params, runs on one A10G
# (g5.xlarge) in minutes and even falls back to CPU for a few hundred rows.
BASE_MODEL = "distilbert-base-uncased"
LABELS = ["Low", "Medium", "High"]
MAX_LEN = 256
SEED = 42
