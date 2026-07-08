# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Agent Bricks — Knowledge Assistant + Supervisor
# MAGIC
# MAGIC Builds two Agent Bricks tiles entirely in code via the Databricks SDK:
# MAGIC
# MAGIC 1. **Knowledge Assistant (KA)** over the claims policy/guideline docs (Volume).
# MAGIC 2. A small **UC function** that looks up structured claims data.
# MAGIC 3. A **Supervisor Agent** that routes between them: policy/procedure questions →
# MAGIC    KA; "look up claim X" / structured questions → the UC function tool.
# MAGIC
# MAGIC Showcases: Agent Bricks (KA + Supervisor) as governed, deployable endpoints, built
# MAGIC and wired together purely in code — no UI clicks.
# MAGIC
# MAGIC > Natural-language SQL over the claims table is shown in the **AI Governance** demo
# MAGIC > via a managed **Genie** MCP server. (The Genie `create_space` SDK path is skipped
# MAGIC > here due to a `serialized_space` serialization bug in the current SDK build.)

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

from config import CATALOG, SCHEMA, CLAIMS_TABLE, VOLUME_PATH
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Knowledge Assistant over the claims docs

# COMMAND ----------

from databricks.sdk.service.knowledgeassistants import (
    KnowledgeAssistant, KnowledgeSource, FilesSpec,
)

ka = w.knowledge_assistants.create_knowledge_assistant(
    knowledge_assistant=KnowledgeAssistant(
        display_name="Claims Knowledge Assistant (demo)",
        description="Answers claims policy and handling-guideline questions.",
        instructions=(
            "You are a claims assistant for a P&C insurer. Answer from the policy "
            "wordings and claims-handling guidelines. Always cite the source document. "
            "If the answer isn't in the docs, say so."
        ),
    )
)
KA_ID, KA_NAME = ka.id, ka.name          # name == 'knowledge-assistants/<id>' (used as parent)
print("KA:", KA_ID, "| endpoint:", ka.endpoint_name)

src = w.knowledge_assistants.create_knowledge_source(
    parent=KA_NAME,
    knowledge_source=KnowledgeSource(
        display_name="Claims policy & guideline docs",
        description="Policy wordings and claims-handling guidelines.",
        source_type="FILES",
        files=FilesSpec(path=VOLUME_PATH),
    ),
)
w.knowledge_assistants.sync_knowledge_sources(name=KA_NAME)
print("Knowledge source added + sync triggered:", src.id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · UC function — structured claims lookup (the Supervisor's data tool)

# COMMAND ----------

CLAIM_LOOKUP_FN = f"{CATALOG}.{SCHEMA}.claim_lookup"
spark.sql(f"""
CREATE OR REPLACE FUNCTION {CLAIM_LOOKUP_FN}(claim_id STRING)
RETURNS TABLE(claim_id STRING, claim_type STRING, peril STRING, claim_status STRING,
              severity STRING, claim_amount DOUBLE, paid_amount DOUBLE, region STRING)
COMMENT 'Look up a single insurance claim by its claim_id (e.g. CLM-100001).'
RETURN SELECT claim_id, claim_type, peril, claim_status, severity, claim_amount, paid_amount, region
       FROM {CLAIMS_TABLE} WHERE claim_id = claim_lookup.claim_id
""")
print("Created UC function:", CLAIM_LOOKUP_FN)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Supervisor Agent routing across KA + UC function
# MAGIC
# MAGIC Note: `tool_type` values are **lowercase** (`knowledge_assistant`, `uc_function`).

# COMMAND ----------

from databricks.sdk.service.supervisoragents import (
    SupervisorAgent, Tool, UcFunction,
    KnowledgeAssistant as SAKnowledgeAssistant,
)

sup = w.supervisor_agents.create_supervisor_agent(
    supervisor_agent=SupervisorAgent(
        display_name="Claims Supervisor (demo)",
        description="Routes claims questions to the right specialist agent.",
        instructions=(
            "Route policy / coverage / handling-procedure questions to the policy_docs "
            "knowledge assistant. When the user asks to look up a specific claim by id "
            "(e.g. CLM-100001) or its status/amount, use the claim_lookup function. "
            "Combine both when a question needs policy context and claim facts."
        ),
    )
)
SUP_NAME = sup.name
print("Supervisor:", sup.supervisor_agent_id or sup.id, "| endpoint:", sup.endpoint_name)

w.supervisor_agents.create_tool(
    parent=SUP_NAME, tool_id="policy_docs",
    tool=Tool(
        tool_type="knowledge_assistant", name="policy_docs",
        description="Policy wordings and claims-handling guidelines.",
        knowledge_assistant=SAKnowledgeAssistant(knowledge_assistant_id=KA_ID),
    ),
)
w.supervisor_agents.create_tool(
    parent=SUP_NAME, tool_id="claim_lookup",
    tool=Tool(
        tool_type="uc_function", name="claim_lookup",
        description="Look up a single claim's type, status, severity, and amounts by claim_id.",
        uc_function=UcFunction(name=CLAIM_LOOKUP_FN),
    ),
)
print("Tools attached: policy_docs (KA) + claim_lookup (UC function)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary — provision takes a few minutes

# COMMAND ----------

print("Knowledge Assistant :", KA_ID, "| endpoint:", ka.endpoint_name)
print("Supervisor          :", sup.supervisor_agent_id or sup.id, "| endpoint:", sup.endpoint_name)
print("\nOpen Agent Bricks in the workspace to chat with the Supervisor, or query the")
print("mas-*-endpoint once it is ACTIVE. Example questions:")
print("  • 'What is the procedure for a basement water damage claim?'  (→ KA)")
print("  • 'Look up claim CLM-100001'                                  (→ claim_lookup)")
