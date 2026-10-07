# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · LoRA fine-tune Qwen3-4B for PII extraction
# MAGIC
# MAGIC Fine-tunes a **LoRA adapter** on the gretel financial-PII data so a small (4B) model matches the
# MAGIC bigger models' extraction quality, then **merges the adapter into the base weights** and snapshots the
# MAGIC merged model to the UC Volume. `02_deploy_endpoints` serves it as a standalone model via the same
# MAGIC vLLM-entrypoint path as the base models (no runtime adapter wiring) — sized to fit an **A100 (GPU_LARGE)**.
# MAGIC
# MAGIC > **Run on Serverless GPU.** A10 works with gradient checkpointing + LoRA; use H100/A100 if you hit OOM.
# MAGIC > Training targets are built from the dataset's ground-truth spans, formatted into the exact output schema.

# COMMAND ----------

# Loose lower-bound pins (exact pins conflict with the serverless-GPU env's constraint file).
# No bitsandbytes: we train in bf16 (no 4-bit). transformers/mlflow come from the base runtime.
# MAGIC %pip install --quiet "peft>=0.12" "trl>=0.12,<0.15" "datasets>=2.20" "accelerate>=0.33" "bitsandbytes>=0.43" hf_transfer
# MAGIC %restart_python

# COMMAND ----------

import sys; sys.path.append("..") if ".." not in sys.path else None
import os, tempfile, json, shutil

os.environ["HF_HOME"] = tempfile.mkdtemp()
os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"]

from config import RAW_TABLE, VOLUME_PATH, LORA_BASE_REPO, LORA_MERGED_DIR, LORA_ADAPTER_DIR
# Train with the COMPACT prompt (no few-shot) so sequences fit an A10; 03 serves the LoRA model with
# the same compact prompt (see pii_common.COMPACT_SYSTEM_PROMPT) to keep train/serve aligned.
from pii_common import COMPACT_SYSTEM_PROMPT, PIIDetectionResult, PIIEntity, PIIType

# Reduce CUDA fragmentation for the loss-logits upcast on A10.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build SFT examples from ground-truth spans
# MAGIC
# MAGIC Each target completion is the correct `PIIDetectionResult` JSON, with `salt_id`s assigned
# MAGIC sequentially per type (the label the model must learn to emit).

# COMMAND ----------

import pandas as pd
from transformers import AutoTokenizer

# Load tokenizer here so we can pre-render the chat template into a "text" column. (TRL's SFTTrainer
# defaults to a "text" field and does not reliably auto-apply the chat template to a "messages"
# column across versions -- pre-rendering is version-robust.)
tokenizer = AutoTokenizer.from_pretrained(LORA_BASE_REPO, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

raw_pdf = spark.table(RAW_TABLE).select("input_text", "ground_truth_json").toPandas()

def build_target(ground_truth):
    """Ground-truth [{type,value}] -> canonical PIIDetectionResult JSON string."""
    counters, entities = {}, []
    for gt in ground_truth:
        t = gt["type"]
        counters[t] = counters.get(t, 0) + 1
        try:
            etype = PIIType(t)
        except ValueError:
            etype = PIIType.OTHER
        entities.append(PIIEntity(entity_type=etype, original_value=gt["value"],
                                  salt_id=f"{t}_{counters[t]:03d}"))
    return PIIDetectionResult(has_pii=len(entities) > 0, entities=entities).model_dump_json()

examples = []
for _, r in raw_pdf.iterrows():
    gt = json.loads(r["ground_truth_json"])
    text = r["input_text"][:2000]
    messages = [
        {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract all PII entities from this document:\n\n{text}"},
        {"role": "assistant", "content": build_target(gt)},
    ]
    examples.append({"text": tokenizer.apply_chat_template(messages, tokenize=False)})

print(f"built {len(examples)} SFT examples")
print("sample text (tail):", examples[0]["text"][-200:])

# COMMAND ----------

from datasets import Dataset
split = int(len(examples) * 0.9)
train_ds = Dataset.from_list(examples[:split])
eval_ds = Dataset.from_list(examples[split:])
print(f"train={len(train_ds)} eval={len(eval_ds)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. LoRA fine-tune

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# QLoRA: load the base in 4-bit (nf4) so a 4B model + LoRA + activations + the vocab-logit loss all
# fit an A10 (24 GB). bf16 full-precision LoRA OOMs here. We merge onto a fresh bf16 base afterwards.
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
# tokenizer already loaded above (used to render the "text" column).
model = AutoModelForCausalLM.from_pretrained(
    LORA_BASE_REPO, quantization_config=bnb, torch_dtype=torch.bfloat16,
    trust_remote_code=True, attn_implementation="eager",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

train_out = tempfile.mkdtemp()
sft_config = SFTConfig(
    output_dir=train_out,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="no",
    dataset_text_field="text",   # pre-rendered chat-template column built above
    max_seq_length=1024,         # compact prompt keeps docs+targets within this; fits A10 logits
    packing=False,
    report_to=[],
)

trainer = SFTTrainer(
    model=model, args=sft_config, peft_config=lora_config,
    train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer,
)
trainer.train()
print("training complete:", trainer.state.log_history[-1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save adapter + merge into base weights → volume snapshot

# COMMAND ----------

# Persist the raw adapter (small; useful for reference / alternate adapter-serving).
adapter_tmp = tempfile.mkdtemp()
trainer.model.save_pretrained(adapter_tmp)
tokenizer.save_pretrained(adapter_tmp)
os.makedirs(LORA_ADAPTER_DIR, exist_ok=True)
shutil.copytree(adapter_tmp, LORA_ADAPTER_DIR, dirs_exist_ok=True)
print("adapter saved ->", LORA_ADAPTER_DIR)

# COMMAND ----------

# Merge: can't merge onto a 4-bit base, so free the training model, reload the base in bf16, apply
# the saved adapter, and merge. The merged bf16 weights are served standalone by notebook 02.
import gc
from peft import PeftModel

del trainer, model
gc.collect(); torch.cuda.empty_cache()

base_bf16 = AutoModelForCausalLM.from_pretrained(
    LORA_BASE_REPO, torch_dtype=torch.bfloat16, trust_remote_code=True,
)
merged = PeftModel.from_pretrained(base_bf16, adapter_tmp).merge_and_unload()
merged_tmp = tempfile.mkdtemp()
merged.save_pretrained(merged_tmp, safe_serialization=True)
tokenizer.save_pretrained(merged_tmp)

# Fresh copy on the volume (overwrite any prior merge).
if os.path.exists(LORA_MERGED_DIR):
    shutil.rmtree(LORA_MERGED_DIR)
os.makedirs(LORA_MERGED_DIR, exist_ok=True)
shutil.copytree(merged_tmp, LORA_MERGED_DIR, dirs_exist_ok=True)
print("merged weights ->", LORA_MERGED_DIR)
print("files:", sorted(os.listdir(LORA_MERGED_DIR))[:8], "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Done
# MAGIC Merged LoRA weights are on the volume. `02_deploy_endpoints` (model key `qwen3-4b-lora`,
# MAGIC `weights_source="volume"`) logs, registers, and serves them exactly like the base models.

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"merged_dir": LORA_MERGED_DIR, "train_examples": len(train_ds)}))
