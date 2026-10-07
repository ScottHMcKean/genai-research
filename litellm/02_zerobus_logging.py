# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · litellm → Zerobus → UC Delta table
# MAGIC
# MAGIC Exercises litellm's **Zerobus logging callback**: every `litellm` request is batched and
# MAGIC POSTed to the Zerobus REST API (`{server_endpoint}/zerobus/v1/tables/<table>/insert`), which
# MAGIC lands it as a row in the managed Delta table created by `00_setup`.
# MAGIC
# MAGIC We drive it **in-process** (no separate proxy server): construct `ZerobusLogger` with the SP
# MAGIC credentials, register it on `litellm.callbacks`, fire a few `acompletion` calls, then force a
# MAGIC flush and read the table back. The callback authenticates as the service principal from 00
# MAGIC (OAuth M2M) — not the notebook user.
# MAGIC
# MAGIC > Prereqs: run `00_setup` first (table + SP + grants + stored secret). Run `01` to confirm
# MAGIC > litellm reaches FMAPI.

# COMMAND ----------

# Clear the serverless PIP_CONSTRAINT so the litellm floor (zerobus needs >=1.103) resolves.
import os
os.environ["PIP_CONSTRAINT"] = ""

# COMMAND ----------

# MAGIC %pip install -U "litellm==1.105.0rc1" -q
# MAGIC %restart_python

# COMMAND ----------

import sys; sys.path.append("..") if ".." not in sys.path else None
import os, asyncio, time
from config import (
    WORKSPACE_URL, ZEROBUS_SERVER_ENDPOINT, LITELLM_TRACES_TABLE, CHAT_MODEL,
    SECRET_SCOPE, SECRET_KEY_CLIENT_ID, SECRET_KEY_CLIENT_SECRET,
)

# SP credentials (stored by 00_setup) and the notebook token for the FMAPI call.
CLIENT_ID = dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY_CLIENT_ID)
CLIENT_SECRET = dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY_CLIENT_SECRET)
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_API_BASE"] = f"{WORKSPACE_URL}/serving-endpoints"
os.environ["DATABRICKS_API_KEY"] = TOKEN

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Register the Zerobus logger
# MAGIC
# MAGIC We construct `ZerobusLogger` explicitly (rather than the `litellm.callbacks = ["zerobus"]`
# MAGIC string form) so the connection params are unambiguous. `start_periodic_flush=False` because
# MAGIC we flush by hand below — deterministic for a notebook with no long-lived event loop.

# COMMAND ----------

import litellm
from litellm.integrations.zerobus.logger import ZerobusLogger
from litellm.types.integrations.zerobus import ZerobusInitParams

params = ZerobusInitParams(
    workspace_url=WORKSPACE_URL,
    server_endpoint=ZEROBUS_SERVER_ENDPOINT,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    table_name=LITELLM_TRACES_TABLE,
    batch_size=100,
    flush_interval=10,
)
zb = ZerobusLogger(params=params, start_periodic_flush=False)
litellm.callbacks = [zb]
print("ZerobusLogger registered -> table:", LITELLM_TRACES_TABLE)
print("insert endpoint:", f"{ZEROBUS_SERVER_ENDPOINT}/zerobus/v1/tables/{LITELLM_TRACES_TABLE}/insert")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fire calls, then flush the batch deterministically

# COMMAND ----------

PROMPTS = [
    "Reply with: Zerobus integration verified.",
    "Name one benefit of Unity Catalog in one sentence.",
    "What is a lakehouse? One sentence.",
]

async def run_and_flush():
    responses = []
    for i, prompt in enumerate(PROMPTS):
        r = await litellm.acompletion(
            model=f"databricks/{CHAT_MODEL}",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            user=f"zerobus-quickstart-{i}",
            metadata={"tags": ["zerobus-quickstart"]},
        )
        responses.append(r)
    # Let the async success callbacks enqueue, then force the batch out to Zerobus.
    await asyncio.sleep(2)
    await zb.flush_queue()
    return responses

# Databricks notebooks already run an asyncio loop, so asyncio.run() would raise. Run the
# coroutine to completion on a dedicated thread with its own fresh loop instead.
def run_coro(coro):
    import threading
    box = {}
    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            box["value"] = loop.run_until_complete(coro)
        except BaseException as e:  # noqa: BLE001 -- surface the real error to the main thread
            box["error"] = e
        finally:
            loop.close()
    t = threading.Thread(target=_runner)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]

responses = run_coro(run_and_flush())
response_ids = [r.id for r in responses]
print("fired", len(responses), "calls")
for r in responses:
    print(" ", r.id, "->", r.choices[0].message.content[:60])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Verify rows landed in Unity Catalog
# MAGIC
# MAGIC Zerobus is at-least-once with an async ingest delay (rows can take ~30–60s to materialize in
# MAGIC Delta), so poll before asserting. We match on the trace `id` (== the litellm response id) —
# MAGIC note the SDK path leaves `end_user` NULL; the `user=` arg surfaces as `request_tags` instead.

# COMMAND ----------

ids_sql = ", ".join(f"'{i}'" for i in response_ids)
this_run = f"id IN ({ids_sql})"

count = 0
for attempt in range(18):  # up to ~90s
    count = spark.table(LITELLM_TRACES_TABLE).filter(this_run).count()
    if count >= len(PROMPTS):
        break
    print(f"attempt {attempt+1}: {count}/{len(PROMPTS)} rows materialized, waiting...")
    time.sleep(5)

print(f"\n{count} of {len(PROMPTS)} rows for this run in {LITELLM_TRACES_TABLE}")
assert count >= 1, (
    "no rows ingested — check SP grants (error 4024 = missing table-level MODIFY/SELECT), "
    "the server endpoint region, and that the OAuth secret in the scope is valid"
)

# COMMAND ----------

display(
    spark.table(LITELLM_TRACES_TABLE)
    .filter(this_run)
    .selectExpr(
        "id", "status", "model", "call_type",
        "prompt_tokens", "completion_tokens", "total_tokens", "response_cost",
        "to_json(request_tags) AS request_tags", "start_time",
    )
    .orderBy("start_time", ascending=False)
)

# COMMAND ----------

print("Zerobus logging verified. Next: 03_gateway_litellm_adapter.")
