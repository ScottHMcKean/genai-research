# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Evaluate & (optionally) serve the fine-tuned triage model
# MAGIC
# MAGIC The punchline: a **small, cheap fine-tuned classifier** can match or beat a
# MAGIC big **zero-shot LLM** on a narrow, high-volume task like claims triage — at a
# MAGIC fraction of the per-call cost and latency.
# MAGIC
# MAGIC 1. Score the **test set** with the fine-tuned UC model → accuracy / macro-F1.
# MAGIC 2. Score the same notes **zero-shot** with the Foundation Model API → accuracy / macro-F1.
# MAGIC 3. Compare, log to MLflow, write a comparison table.
# MAGIC 4. *(optional)* Deploy the fine-tuned model to a **Model Serving** endpoint.
# MAGIC
# MAGIC Runs on **serverless** (base env `4`/`5`); no GPU needed for inference on DistilBERT.

# COMMAND ----------

# MAGIC %pip install --quiet "transformers>=4.44" torch scikit-learn openai mlflow
# MAGIC %restart_python

# COMMAND ----------

import sys, os
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".")
from config import (TEST_TABLE, UC_MODEL, SERVING_ENDPOINT, CHAT_MODEL, EVAL_TABLE, LABELS)

import pandas as pd, mlflow
from sklearn.metrics import accuracy_score, f1_score

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/ai_runtime_claims")
mlflow.set_registry_uri("databricks-uc")

test_pd = spark.table(TEST_TABLE).select("claim_id", "note", "severity").toPandas()
# Cap the zero-shot baseline for a snappy demo; fine-tuned scores the full set.
BASELINE_N = min(200, len(test_pd))
print(f"test rows: {len(test_pd)} | zero-shot baseline on: {BASELINE_N}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Fine-tuned model (from Unity Catalog)

# COMMAND ----------

# Load the latest registered version so a re-run of 02 is always what's evaluated/served.
from mlflow.tracking import MlflowClient
mlflow.set_registry_uri("databricks-uc")
FT_VERSION = max(int(v.version) for v in MlflowClient().search_model_versions(f"name='{UC_MODEL}'"))
print("Evaluating fine-tuned model version:", FT_VERSION)
ft = mlflow.pyfunc.load_model(f"models:/{UC_MODEL}/{FT_VERSION}")
ft_pred = ft.predict(test_pd[["note"]])["pred_severity"].tolist()
ft_acc = accuracy_score(test_pd["severity"], ft_pred)
ft_f1 = f1_score(test_pd["severity"], ft_pred, average="macro", labels=LABELS)
print(f"Fine-tuned  -> accuracy {ft_acc:.3f} | macro-F1 {ft_f1:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call the fine-tuned model directly — no serving endpoint needed
# MAGIC `ft` is the loaded model artifact; predict on any note in-process.

# COMMAND ----------

demo_notes = pd.DataFrame({"note": [
    "Multi-vehicle collision on the highway, driver hospitalized, vehicle a likely total loss.",
    "Small hail dents on the hood, cosmetic only, vehicle drivable.",
]})
demo_notes["pred_severity"] = ft.predict(demo_notes)["pred_severity"]
display(demo_notes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Zero-shot Foundation Model API baseline (same notes)

# COMMAND ----------

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
BASE_URL = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}/serving-endpoints"
client = OpenAI(api_key=TOKEN, base_url=BASE_URL)

PROMPT = ("Classify P&C claim severity as exactly one of Low, Medium, High. "
          "Low=cosmetic/minor; Medium=moderate/minor injury; High=total loss/hospitalization/"
          "litigation/uninhabitable. Respond with ONLY the word.\n\nNote: ")

def zshot(note):
    try:
        r = client.chat.completions.create(
            model=CHAT_MODEL, max_tokens=4, temperature=0,
            messages=[{"role": "user", "content": PROMPT + note}])
        p = r.choices[0].message.content.strip().split()[0].capitalize()
        return p if p in LABELS else "Medium"
    except Exception:
        return "Medium"

base_df = test_pd.head(BASELINE_N).copy()
with ThreadPoolExecutor(max_workers=16) as ex:
    base_df["zs_pred"] = list(ex.map(zshot, base_df["note"].tolist()))

zs_acc = accuracy_score(base_df["severity"], base_df["zs_pred"])
zs_f1 = f1_score(base_df["severity"], base_df["zs_pred"], average="macro", labels=LABELS)
print(f"Zero-shot   -> accuracy {zs_acc:.3f} | macro-F1 {zs_f1:.3f}  (n={BASELINE_N})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Compare + log

# COMMAND ----------

comparison = pd.DataFrame([
    {"approach": "fine_tuned_distilbert_lora", "accuracy": round(ft_acc, 3),
     "f1_macro": round(ft_f1, 3), "n": len(test_pd)},
    {"approach": f"zero_shot_{CHAT_MODEL}", "accuracy": round(zs_acc, 3),
     "f1_macro": round(zs_f1, 3), "n": BASELINE_N},
])
spark.createDataFrame(comparison).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(EVAL_TABLE)

with mlflow.start_run(run_name="triage_ft_vs_zeroshot"):
    mlflow.log_metric("ft_accuracy", float(ft_acc)); mlflow.log_metric("ft_f1_macro", float(ft_f1))
    mlflow.log_metric("zs_accuracy", float(zs_acc)); mlflow.log_metric("zs_f1_macro", float(zs_f1))
    mlflow.log_table(comparison, "comparison.json")

print("Wrote", EVAL_TABLE)
display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3b · Populate traces — 10 triage examples
# MAGIC
# MAGIC Runs 10 sample notes through the fine-tuned model and the zero-shot LLM inside MLflow
# MAGIC spans, so the Traces UI shows both paths side by side for the triage task.

# COMMAND ----------

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/claims_triage")

sample = test_pd.head(10)[["note", "severity"]].to_dict("records")
for i, row in enumerate(sample, 1):
    with mlflow.start_span(name=f"triage_{i:02d}") as span:
        span.set_inputs({"note": row["note"]})
        ft_p = ft.predict(pd.DataFrame([{"note": row["note"]}]))["pred_severity"].iloc[0]
        zs_p = zshot(row["note"])
        span.set_outputs({"fine_tuned": ft_p, "zero_shot": zs_p, "label": row["severity"]})
print(f"Wrote 10 triage traces to /Users/{user}/claims_triage")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · (Optional) Deploy to Model Serving
# MAGIC
# MAGIC DistilBERT is small → CPU serving is fine and cheap. Set `DEPLOY = True` to create
# MAGIC the endpoint. Left **off** by default so the demo doesn't provision (and bill for) an
# MAGIC endpoint unattended.

# COMMAND ----------

DEPLOY = False  # flip to True to actually create the serving endpoint

if DEPLOY:
    from mlflow.deployments import get_deploy_client
    dc = get_deploy_client("databricks")
    dc.create_endpoint(
        name=SERVING_ENDPOINT,
        config={"served_entities": [{
            "entity_name": UC_MODEL, "entity_version": str(FT_VERSION),
            "workload_size": "Small", "scale_to_zero_enabled": True,
        }]},
    )
    print("Creating endpoint", SERVING_ENDPOINT, "-- provisions in a few minutes.")
else:
    print("DEPLOY=False. Fine-tuned model is registered in UC at", UC_MODEL,
          "\nSet DEPLOY=True to serve it behind", SERVING_ENDPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Talk track
# MAGIC A ~66M-param model you fine-tuned in minutes competes with a frontier LLM on this
# MAGIC narrow task — but costs orders of magnitude less per claim and runs in milliseconds.
# MAGIC For high-volume, repetitive decisions (triage, routing, classification), **fine-tune
# MAGIC small**; save the big LLM for open-ended reasoning. Same platform, same governance.
