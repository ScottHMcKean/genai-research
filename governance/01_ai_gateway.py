# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Unity AI Gateway — govern a claims-facing LLM
# MAGIC
# MAGIC One control plane for every model call. On a serving endpoint we turn on:
# MAGIC
# MAGIC | Feature | What you get |
# MAGIC |---|---|
# MAGIC | **Usage tracking** | Per-request tokens / latency / cost in system tables |
# MAGIC | **Inference tables** | Full request+response payloads logged to a UC Delta table (audit) |
# MAGIC | **Rate limits** | Cap calls per user / endpoint to control spend & abuse |
# MAGIC | **Guardrails — PII** | Detect & **mask** PII (e.g. a claimant SSN) on input **and** output |
# MAGIC
# MAGIC The same RBAC that governs the claims table governs the model. Nothing in the agent
# MAGIC code changes — governance is applied at the gateway.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-sdk openai
# MAGIC %restart_python

# COMMAND ----------

import sys, os
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".")
from config import (CATALOG, GATEWAY_INFERENCE_SCHEMA, GATEWAY_ENDPOINT, CHAT_MODEL)

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayGuardrails, AiGatewayGuardrailParameters,
    AiGatewayGuardrailPiiBehavior, AiGatewayGuardrailPiiBehaviorBehavior,
    AiGatewayInferenceTableConfig, AiGatewayUsageTrackingConfig,
    AiGatewayRateLimit, AiGatewayRateLimitKey, AiGatewayRateLimitRenewalPeriod,
    ExternalModel, ExternalModelProvider, DatabricksModelServingConfig,
    EndpointCoreConfigInput, ServedEntityInput,
)

w = WorkspaceClient()
host = spark.conf.get("spark.databricks.workspaceUrl")
print("Workspace:", host)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · An endpoint we own (so we can attach a gateway)
# MAGIC
# MAGIC Pay-per-token foundation-model endpoints (`databricks-claude-sonnet-4-5`) are
# MAGIC system-managed, so we stand up **our own** endpoint that proxies the same model via
# MAGIC an *external model* route. AI Gateway attaches to endpoints you own.
# MAGIC
# MAGIC > The external-model route needs a token to call the in-workspace FM API. Prefer a
# MAGIC > **secret-backed** token (a service-principal PAT) so the endpoint keeps working after
# MAGIC > your session ends — set `TOKEN_SECRET_SCOPE`/`TOKEN_SECRET_KEY` in `config.py`. The
# MAGIC > notebook token is only a last-resort fallback and **expires with the session**, after
# MAGIC > which the gateway route returns 401 until re-run. (You can also skip this cell and
# MAGIC > attach the gateway in cell 2 to any endpoint you already own.)

# COMMAND ----------

from config import TOKEN_SECRET_SCOPE, TOKEN_SECRET_KEY

base_url = f"https://{host}/serving-endpoints"
if TOKEN_SECRET_SCOPE and TOKEN_SECRET_KEY:
    token = dbutils.secrets.get(TOKEN_SECRET_SCOPE, TOKEN_SECRET_KEY)   # durable (SP PAT)
    print("Using secret-backed token for the external-model route.")
else:
    # Fallback: ephemeral notebook token -- the gateway route will 401 once it expires.
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    print("WARNING: using the ephemeral notebook token; the gateway route will fail auth "
          "after this session ends. Set TOKEN_SECRET_SCOPE/KEY in config.py for a durable token.")

existing = [e.name for e in w.serving_endpoints.list()]
if GATEWAY_ENDPOINT not in existing:
    print("Creating", GATEWAY_ENDPOINT, "...")
    w.serving_endpoints.create(
        name=GATEWAY_ENDPOINT,
        config=EndpointCoreConfigInput(
            name=GATEWAY_ENDPOINT,
            served_entities=[
                ServedEntityInput(
                    name="claims-llm",
                    external_model=ExternalModel(
                        name=CHAT_MODEL,
                        provider=ExternalModelProvider.DATABRICKS_MODEL_SERVING,
                        task="llm/v1/chat",
                        # Proxy the in-workspace pay-per-token FM endpoint.
                        databricks_model_serving_config=DatabricksModelServingConfig(
                            databricks_workspace_url=f"https://{host}",
                            databricks_api_token_plaintext=token,
                        ),
                    ),
                )
            ]
        ),
    )
    print("Created (provisioning).")
else:
    print(GATEWAY_ENDPOINT, "already exists — reusing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Attach the AI Gateway config
# MAGIC All four governance features in one call.

# COMMAND ----------

w.serving_endpoints.put_ai_gateway(
    name=GATEWAY_ENDPOINT,
    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
    inference_table_config=AiGatewayInferenceTableConfig(
        enabled=True,
        catalog_name=CATALOG,
        schema_name=GATEWAY_INFERENCE_SCHEMA,
        table_name_prefix="gateway_claims_llm",   # -> claims_demo.gateway_claims_llm_payload
    ),
    rate_limits=[
        AiGatewayRateLimit(
            calls=100,
            renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE,
            key=AiGatewayRateLimitKey.USER,       # 100 calls/user/min
        )
    ],
    guardrails=AiGatewayGuardrails(
        input=AiGatewayGuardrailParameters(
            pii=AiGatewayGuardrailPiiBehavior(
                behavior=AiGatewayGuardrailPiiBehaviorBehavior.MASK
            ),
        ),
        output=AiGatewayGuardrailParameters(
            pii=AiGatewayGuardrailPiiBehavior(
                behavior=AiGatewayGuardrailPiiBehaviorBehavior.MASK
            ),
        ),
    ),
)
print("AI Gateway configured on", GATEWAY_ENDPOINT)
print(" - usage tracking: on")
print(" - inference table:", f"{CATALOG}.{GATEWAY_INFERENCE_SCHEMA}.gateway_claims_llm_payload")
print(" - rate limit: 100 calls/user/min")
print(" - PII guardrail: MASK (input + output)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Show the PII guardrail in action
# MAGIC Send an adjuster note that contains a claimant SSN; the gateway masks it before the
# MAGIC prompt ever reaches the model (and again on the way out).

# COMMAND ----------

from openai import OpenAI

client = OpenAI(api_key=token, base_url=base_url)
prompt = (
    "Summarize this claim note for the file. "
    "Claimant John Smith, SSN 123-45-6789, reported a rear-end collision on the 401; "
    "moderate rear-quarter damage, vehicle towed for assessment."
)
try:
    resp = client.chat.completions.create(
        model=GATEWAY_ENDPOINT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    print(resp.choices[0].message.content)
    print("\n^ Note the SSN is masked -- the guardrail redacted it at the gateway.")
except Exception as e:
    print("Query failed (endpoint may still be provisioning, or token/route restricted):", e)
    print("Fallback: query in the UI (Serving > claims-llm-gateway > Use) or wait for READY.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3b · Populate traces / audit log — 10 questions through the gateway
# MAGIC Sends 10 realistic claims questions so usage metrics and the inference table (below)
# MAGIC have content, and the MLflow Traces UI shows governed gateway calls.

# COMMAND ----------

QUESTIONS = [
    "What is the procedure for a basement water damage claim?",
    "Is sewer backup covered under the homeowners policy?",
    "When can a glass-only auto claim be settled without an inspection?",
    "What fraud indicators require a Special Investigations Unit referral?",
    "What triggers escalation to a senior adjuster?",
    "How are Additional Living Expenses handled after a house fire?",
    "What is the insured's duty to mitigate after water damage?",
    "When should an auto claim be declared a total loss?",
    "What is the catastrophe (CAT) response procedure?",
    "How is a stolen financed vehicle settled?",
]
ok = 0
for i, q in enumerate(QUESTIONS, 1):
    try:
        client.chat.completions.create(
            model=GATEWAY_ENDPOINT,
            messages=[{"role": "user", "content": q}],
            max_tokens=200,
        )
        ok += 1
    except Exception as e:
        print(f"[{i:02d}] skipped: {str(e)[:80]}")
print(f"Sent {ok}/{len(QUESTIONS)} questions through {GATEWAY_ENDPOINT} "
      f"(usage + inference table populate asynchronously).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Where the audit trail lands
# MAGIC Inference-table logging is asynchronous (a few minutes). Once populated, every
# MAGIC request+response is a governed Delta row.

# COMMAND ----------

payload_table = f"{CATALOG}.{GATEWAY_INFERENCE_SCHEMA}.gateway_claims_llm_payload"
try:
    df = spark.table(payload_table)
    print(f"{payload_table}: {df.count()} rows logged")
    display(df.orderBy("timestamp_ms" if "timestamp_ms" in df.columns else df.columns[0]).limit(5))
except Exception as e:
    print(f"{payload_table} not populated yet (logging lags a few minutes). {type(e).__name__}")
    print("Re-run this cell after sending a few requests.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Talk track
# MAGIC - **One gateway, every model.** Swap Claude for GPT or an internal model — governance
# MAGIC   (limits, logging, guardrails) stays put; agent code is untouched.
# MAGIC - **PII never leaves the perimeter unmasked** — enforced server-side, not in prompts.
# MAGIC - **Auditable by construction** — every call is a UC Delta row, governed by the same
# MAGIC   Unity Catalog permissions as the claims data itself.
