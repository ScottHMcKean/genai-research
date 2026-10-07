# Config for the PII-scrubbing vLLM-entrypoint serving demo.
# Mirrors agents/config.py -- kept local so the folder is self-contained when synced as a bundle.
#
# Governance flow (the whole point of the two catalogs):
#   RAW  PII  ->  shm_skunkworks_catalog  (workspace-local; restricted)
#   REDACTED  ->  shm_catalog             (shared metastore; masked text + metadata only)
# Raw PII never crosses into the shared catalog.

from databricks.sdk.service.serving import ServingModelWorkloadType

# --- Catalogs / schema ---
CATALOG_RAW = "shm_skunkworks_catalog"   # workspace catalog: raw PII lands here
CATALOG_REDACTED = "shm_catalog"         # shared metastore catalog: only redacted data
SCHEMA = "pii"

# --- Tables ---
RAW_TABLE = f"{CATALOG_RAW}.{SCHEMA}.raw_documents"            # raw PII documents (input)
REDACTED_TABLE = f"{CATALOG_REDACTED}.{SCHEMA}.redacted_documents"  # masked output (shared)
RESULTS_TABLE = f"{CATALOG_RAW}.{SCHEMA}.scrub_results"        # per-model bench/quality rows

# --- HuggingFace weight cache (UC Volume) ---
VOLUME = "huggingface"
VOLUME_PATH = f"/Volumes/{CATALOG_RAW}/{SCHEMA}/{VOLUME}"

# --- HF token secret (Gemma is license-gated) ---
#   databricks secrets create-scope shm
#   databricks secrets put-secret shm hf_token
HF_SECRET_SCOPE = "shm"
HF_SECRET_KEY = "hf_token"

# --- MLflow experiment ---
EXPERIMENT_NAME = "pii-serving-sweep"

# --- Dataset ---
GRETEL_DATASET = "gretelai/synthetic_pii_finance_multilingual"
NUM_ROWS = 500           # docs to land / scrub
MAX_DOC_CHARS = 3000     # dataset filter (English docs under this length)

# --- LoRA fine-tune (Qwen3-4B) ---
LORA_BASE_REPO = "Qwen/Qwen3-4B"
LORA_MERGED_DIR = f"{VOLUME_PATH}/qwen3-4b-pii-lora-merged"   # merged weights snapshot (served as standalone)
LORA_ADAPTER_DIR = f"{VOLUME_PATH}/qwen3-4b-pii-lora-adapter"  # raw adapter (kept for reference)

# --- Model lineup (the 3-model sweep) ---
# workload_type: string kept here, resolved to ServingModelWorkloadType in the deploy notebook.
#   GPU_MEDIUM = A10 (24 GB) -- fits the 4B base
#   GPU_LARGE  = A100 (80 GB) -- required by Gemma-27B and used for the LoRA model per the ask
MODELS = {
    "qwen3-4b": {
        "hf_repo": "Qwen/Qwen3-4B",
        "uc_model": f"{CATALOG_RAW}.{SCHEMA}.pii_qwen3_4b",
        "endpoint": "shm-pii-qwen3-4b",
        "workload_type": "GPU_MEDIUM",
        "gated": False,
        "weights_source": "hf",             # download from HF -> volume snapshot
    },
    "gemma3-27b": {
        "hf_repo": "google/gemma-3-27b-it",
        "uc_model": f"{CATALOG_RAW}.{SCHEMA}.pii_gemma3_27b",
        "endpoint": "shm-pii-gemma3-27b",
        "workload_type": "GPU_LARGE",
        "gated": True,
        "weights_source": "hf",
    },
    "qwen3-4b-lora": {
        "hf_repo": None,                    # served from LORA_MERGED_DIR, produced by 01_lora_finetune_qwen
        "base_repo": "Qwen/Qwen3-4B",
        "uc_model": f"{CATALOG_RAW}.{SCHEMA}.pii_qwen3_4b_lora",
        "endpoint": "shm-pii-qwen3-4b-lora",
        "workload_type": "GPU_LARGE",
        "gated": False,
        "weights_source": "volume",         # merged weights already on the volume
        "weights_dir": LORA_MERGED_DIR,
    },
}

# String -> SDK enum for the deploy notebook.
WORKLOAD_TYPES = {
    "GPU_SMALL": ServingModelWorkloadType.GPU_SMALL,     # T4
    "GPU_MEDIUM": ServingModelWorkloadType.GPU_MEDIUM,   # A10
    "GPU_LARGE": ServingModelWorkloadType.GPU_LARGE,     # A100 80 GB
}

# --- Serving endpoint scaling ---
# Requested: scale-to-zero (idle endpoints cost nothing). NOTE: the Custom LLM Serving *entrypoint*
# route has historically REJECTED autoscaling/scale-to-zero ("Served entity with entrypoint does not
# support autoscaling"). 02_deploy_endpoints attempts this value and falls back to fixed concurrency
# if the platform rejects it.
SCALE_TO_ZERO_ENABLED = True

# --- vLLM / serving tuning ---
DTYPE = "bfloat16"
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.85
LOCAL_PORT = 3080        # serverless GPU notebooks allow 3000-3999
SERVING_PORT = 8080      # Model Serving requires 8080

# --- Pinned deps (match the tested Custom LLM Serving starter versions) ---
SERVING_PIP = (
    "vllm==0.11.2 transformers==4.57.6 openai==2.17.0 "
    "opencv-python-headless==4.12.* mlflow==3.12.0 hf_transfer==0.1.9 databricks-sdk>=0.102.0"
)
TRAIN_PIP = (
    # Loose lower bounds -- exact pins conflict with the serverless-GPU env constraint file.
    "'peft>=0.12' 'trl>=0.12,<0.15' 'datasets>=2.20' 'accelerate>=0.33' hf_transfer"
)
