# AI Governance — Unity AI Gateway + MCP + Observability

Governance for AI on Databricks, on the shared **insurance-claims** spine. The theme: the
*same* Unity Catalog that governs the claims data governs the models, tools, and agents —
so security, PII protection, cost, and lineage come for free, not as bolt-ons.

Maps to the Day-2 section **"MCP + AI Gateway"**.

## Prerequisites

Run **`claims_demo/00_setup_and_data.py`** once — it builds
`shm_skunkworks_catalog.claims_demo` (`claims` with synthetic PII, `claim_notes`,
`doc_chunks`, `docs` volume). Everything here reads that.

## Notebooks (run in order)

| # | Notebook | What it shows |
|---|----------|---------------|
| 00 | `00_setup.py` | Verifies the shared claims spine exists (no data regen). |
| 01 | `01_ai_gateway.py` | **Unity AI Gateway** on an endpoint we own: usage tracking, **inference tables** (payload audit), **rate limits**, and **PII guardrails** that mask a claimant SSN on input *and* output. |
| 02 | `02_mcp_governance.py` | **MCP** on Databricks: two claims **UC functions** exposed as a **managed MCP server**, on-behalf-of-user auth, and the UC-connection pattern for **external** MCP servers. |
| 03 | `03_governance_observability.py` | **System-table** usage/token telemetry, **cost attribution** (`system.billing`), the gateway **payload log**, and **UC lineage** from claims data → AI. |

## Talk track

- **One gateway, every model** — swap Claude for GPT or an internal model; limits, logging,
  and guardrails stay put and agent code is untouched.
- **PII never leaves unmasked** — enforced server-side at the gateway, not in prompts.
- **Your governed assets are the tools** — UC functions / Vector Search / Genie become MCP
  tools with lineage and RBAC; agents run with the *caller's* permissions.
- **Auditable + chargeback-ready** — every call is a governed Delta row; spend and lineage
  are one SQL query away (the questions model-risk teams always ask).

## Notes & cost

- Foundation-model calls are **pay-per-token** (cents for the demo).
- The gateway endpoint (`claims-llm-gateway`) proxies the in-workspace FM API via the
  `DATABRICKS_MODEL_SERVING` external-model provider using the notebook's own token. If your
  workspace restricts token creation, attach the identical gateway config (01, cell 2) to any
  endpoint you already own — the governance story is the same.
- Inference-table and system-table telemetry are **asynchronous** (minutes to a few hours);
  re-run notebook 03 later to see populated rows.

## Deploy

```bash
databricks bundle deploy -t dev
databricks bundle run   -t dev ai_governance_job
```
