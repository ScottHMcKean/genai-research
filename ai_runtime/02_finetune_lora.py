# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · LoRA fine-tune — claims triage classifier
# MAGIC
# MAGIC A **real, lightweight LoRA fine-tune**: adapt `distilbert-base-uncased` (a ~66M-param
# MAGIC encoder) into a 3-class **claims-severity** classifier (Low / Medium / High) from the
# MAGIC adjuster note text. Logged to MLflow and registered in **Unity Catalog**.
# MAGIC
# MAGIC Small model + small data on purpose → finishes in **minutes**.
# MAGIC
# MAGIC ## Compute
# MAGIC - **Preferred:** a **GPU** (this notebook auto-detects CUDA). On this workspace a
# MAGIC   single **g5.xlarge** (A10G) is ideal — see the DAB job's commented GPU cluster.
# MAGIC - **Serverless GPU** (base env `5`, if enabled on the workspace) also works.
# MAGIC - **Fallback:** runs on **CPU** too — DistilBERT LoRA over a few hundred short
# MAGIC   notes trains on CPU in a few minutes. So the demo never hard-blocks on GPU.

# COMMAND ----------

# MAGIC %pip install --quiet "transformers>=4.44" "peft>=0.12" "datasets>=2.20" "accelerate>=0.33" torch scikit-learn mlflow
# MAGIC %restart_python

# COMMAND ----------

import sys, os
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".")
from config import (TRAIN_TABLE, TEST_TABLE, BASE_MODEL, UC_MODEL, LABELS, MAX_LEN, SEED)

# Databricks serverless sets torch-distributed env vars (RANK/WORLD_SIZE/MASTER_ADDR/...).
# HF Trainer reads them and tries to init a DDP process group that never connects on a
# single serverless context -- it hangs ~30 min then raises DistNetworkError. Clear them
# so training runs single-process.
for _v in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
           "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "TORCHELASTIC_RUN_ID"):
    os.environ.pop(_v, None)

import numpy as np, pandas as pd, torch, mlflow
print("CUDA available:", torch.cuda.is_available(),
      "| device:", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/ai_runtime_claims")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Load the train/test split (from `00_setup`)

# COMMAND ----------

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

train_pd = spark.table(TRAIN_TABLE).select("note", "severity").toPandas()
test_pd = spark.table(TEST_TABLE).select("note", "severity").toPandas()

# This FEVM is serverless-only (no GPU), so training runs on CPU. Cap the training
# set for demo speed -- the point is to show a real LoRA fine-tune loop, not to
# maximize accuracy. Remove the cap (or run on serverless GPU) for a full run.
if DEVICE == "cpu":
    # CPU on serverless is the slow path -- shrink hard so the LoRA loop finishes in a
    # few minutes for a live demo. The point is a real fine-tune, not peak accuracy.
    train_pd = train_pd.sample(n=min(150, len(train_pd)), random_state=SEED).reset_index(drop=True)
    test_pd = test_pd.sample(n=min(120, len(test_pd)), random_state=SEED).reset_index(drop=True)
print(f"train rows: {len(train_pd)} | test rows: {len(test_pd)}")
train_pd["label"] = train_pd["severity"].map(label2id)
test_pd["label"] = test_pd["severity"].map(label2id)
print(f"train: {len(train_pd)} | test: {len(test_pd)}")
print(train_pd["severity"].value_counts().to_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Tokenize

# COMMAND ----------

from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def to_ds(df):
    ds = Dataset.from_pandas(df[["note", "label"]], preserve_index=False)
    return ds.map(lambda b: tokenizer(b["note"], truncation=True, max_length=MAX_LEN),
                  batched=True)

train_ds, test_ds = to_ds(train_pd), to_ds(test_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Wrap the base model with a LoRA adapter
# MAGIC
# MAGIC Only the small adapter matrices train; the 66M base weights stay frozen — that's
# MAGIC the whole point of LoRA (cheap, fast, tiny artifact).

# COMMAND ----------

from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

base = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id)

lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_lin", "v_lin"],   # DistilBERT attention projections
)
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Train

# COMMAND ----------

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score

def metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro")}

out_dir = "/tmp/claims_triage_lora"
args = TrainingArguments(
    output_dir=out_dir, num_train_epochs=(1 if DEVICE == "cpu" else 4),
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    learning_rate=2e-4, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="no", logging_steps=20,
    seed=SEED, report_to=[], use_cpu=(DEVICE == "cpu"),
)
import inspect
_tk = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
           data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
# transformers >=4.46 renamed the `tokenizer` arg to `processing_class`.
if "processing_class" in inspect.signature(Trainer.__init__).parameters:
    _tk["processing_class"] = tokenizer
else:
    _tk["tokenizer"] = tokenizer
trainer = Trainer(**_tk)
mlflow.autolog(disable=True)  # we log explicitly below
train_result = trainer.train()
eval_result = trainer.evaluate()
print("Eval:", eval_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Merge adapter → save a self-contained model
# MAGIC
# MAGIC `merge_and_unload()` folds the LoRA weights into the base so the serving artifact is
# MAGIC a plain Transformers model (serving only needs `transformers` + `torch`, not `peft`).

# COMMAND ----------

merged = model.merge_and_unload()
save_dir = "/tmp/claims_triage_merged"
merged.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Saved merged model to", save_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Log to MLflow + register in Unity Catalog
# MAGIC
# MAGIC A small pyfunc wrapper loads the merged Transformers model and returns a severity
# MAGIC label per input note — so it serves behind a normal Model Serving endpoint.

# COMMAND ----------

class ClaimsTriageModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.torch = torch
        d = context.artifacts["model_dir"]
        self.tok = AutoTokenizer.from_pretrained(d)
        self.model = AutoModelForSequenceClassification.from_pretrained(d)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, context, model_input):
        import pandas as pd
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].astype(str).tolist()
        else:
            texts = [str(x) for x in model_input]
        out = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            enc = self.tok(batch, truncation=True, max_length=256,
                           padding=True, return_tensors="pt")
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            for idx in logits.argmax(-1).tolist():
                out.append(self.id2label[idx])
        return pd.DataFrame({"pred_severity": out})

from mlflow.models import infer_signature
sig = infer_signature(pd.DataFrame({"note": ["Minor windshield chip, glass-only."]}),
                      pd.DataFrame({"pred_severity": ["Low"]}))

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_name="lora_finetune_triage") as run:
    mlflow.log_param("base_model", BASE_MODEL)
    mlflow.log_param("method", "LoRA (r=8, q_lin/v_lin)")
    mlflow.log_param("device", DEVICE)
    mlflow.log_param("epochs", 4)
    mlflow.log_metric("test_accuracy", float(eval_result["eval_accuracy"]))
    mlflow.log_metric("test_f1_macro", float(eval_result["eval_f1_macro"]))
    mlflow.log_metric("train_runtime_s", float(train_result.metrics.get("train_runtime", 0.0)))
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ClaimsTriageModel(),
        artifacts={"model_dir": save_dir},
        signature=sig,
        input_example=pd.DataFrame({"note": ["Multi-vehicle collision, driver hospitalized."]}),
        pip_requirements=["transformers>=4.44", "torch", "pandas"],
        registered_model_name=UC_MODEL,
    )
print("Logged + registered:", UC_MODEL)
print("run_id:", run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick sanity check on the logged model

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(info.model_uri)
samples = pd.DataFrame({"note": [
    "Cracked windshield from road debris, glass-only claim.",
    "House fire, two floors uninhabitable, ALE and rebuild required.",
    "Rear-end collision, moderate rear-quarter damage, towed for assessment.",
]})
display(pd.concat([samples, loaded.predict(samples)], axis=1))

# COMMAND ----------

# MAGIC %md
# MAGIC Fine-tuned classifier is in Unity Catalog. Next: `03_serve_and_eval` compares it
# MAGIC against the zero-shot FMAPI baseline and (optionally) serves it.
