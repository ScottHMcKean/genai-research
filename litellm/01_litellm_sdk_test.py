# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Test the litellm package against Databricks FMAPI
# MAGIC
# MAGIC Smoke-tests the **litellm** Python SDK using the `databricks/` provider, pointed at this
# MAGIC workspace's Foundation Model API serving endpoints. No Zerobus yet — just proving litellm
# MAGIC talks to Databricks and reports usage/cost. Covers:
# MAGIC 1. A blocking `completion` call.
# MAGIC 2. A streaming call.
# MAGIC 3. Token usage + `response_cost` (litellm's cost tracker).
# MAGIC 4. An optional `embedding` call.
# MAGIC
# MAGIC litellm's `databricks/` provider reads `DATABRICKS_API_BASE` (the `/serving-endpoints` root)
# MAGIC and `DATABRICKS_API_KEY`. We use the notebook's own token so no PAT is needed.

# COMMAND ----------

# Clear the serverless PIP_CONSTRAINT so the litellm floor resolves (see 00_setup / config.py).
import os
os.environ["PIP_CONSTRAINT"] = ""

# COMMAND ----------

# MAGIC %pip install -U "litellm==1.105.0rc1" -q
# MAGIC %restart_python

# COMMAND ----------

import sys; sys.path.append("..") if ".." not in sys.path else None
import os
from config import WORKSPACE_URL, CHAT_MODEL, EMBED_MODEL

# Notebook-native token -> litellm databricks provider.
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_API_BASE"] = f"{WORKSPACE_URL}/serving-endpoints"
os.environ["DATABRICKS_API_KEY"] = TOKEN

import litellm
from importlib.metadata import version as _pkgver
print("litellm version:", _pkgver("litellm"))
print("api_base       :", os.environ["DATABRICKS_API_BASE"])
print("chat model     :", CHAT_MODEL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Blocking completion

# COMMAND ----------

resp = litellm.completion(
    model=f"databricks/{CHAT_MODEL}",
    messages=[{"role": "user", "content": "Reply with exactly: litellm on Databricks verified."}],
    max_tokens=50,
    user="litellm-sdk-test",
    metadata={"tags": ["litellm-sdk-test"]},
)
print("id      :", resp.id)
print("model   :", resp.model)
print("content :", resp.choices[0].message.content)
print("usage   :", resp.usage)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Streaming

# COMMAND ----------

stream = litellm.completion(
    model=f"databricks/{CHAT_MODEL}",
    messages=[{"role": "user", "content": "Count from 1 to 5, space-separated."}],
    max_tokens=50,
    stream=True,
)
chunks = []
for chunk in stream:
    piece = chunk.choices[0].delta.content or ""
    chunks.append(piece)
    print(piece, end="")
print("\n[streamed", len(chunks), "chunks]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Usage + cost tracking
# MAGIC
# MAGIC `completion_cost` is litellm's own estimate from its model-price map. Databricks pay-per-token
# MAGIC models may not be in that map — a $0.0 / "not found" result just means litellm has no price
# MAGIC entry, not that the call failed.

# COMMAND ----------

try:
    cost = litellm.completion_cost(completion_response=resp)
    print(f"litellm-estimated cost: ${cost:.6f}")
except Exception as e:  # noqa: BLE001 -- model may be absent from litellm's price map
    print("completion_cost unavailable for this model:", e)

print("prompt_tokens    :", resp.usage.prompt_tokens)
print("completion_tokens:", resp.usage.completion_tokens)
print("total_tokens     :", resp.usage.total_tokens)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Embeddings (optional)
# MAGIC
# MAGIC Skips cleanly if the embedding endpoint isn't served on this workspace.

# COMMAND ----------

try:
    emb = litellm.embedding(
        model=f"databricks/{EMBED_MODEL}",
        input=["Databricks lakehouse", "LiteLLM gateway"],
    )
    print(f"{EMBED_MODEL}: {len(emb.data)} vectors, dim={len(emb.data[0]['embedding'])}")
except Exception as e:  # noqa: BLE001 -- embedding endpoint optional
    print(f"embedding test skipped ({EMBED_MODEL} not available): {str(e)[:160]}")

# COMMAND ----------

print("litellm SDK smoke test complete. Next: 02_zerobus_logging.")
