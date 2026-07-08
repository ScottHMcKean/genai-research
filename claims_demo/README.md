# Claims Demo — shared data spine

The **one insurance-and-claims dataset** the three demos build on. Run this first;
it's the "build from scratch" step and is idempotent (safe to re-run).

## What it creates (`shm_skunkworks_catalog.claims_demo`)

| Object | Description | Feeds |
|--------|-------------|-------|
| `claims` | ~2,000 synthetic P&C claims — incl. **synthetic PII** (`customer_name`, `customer_email`, `customer_ssn`), `claim_type`, `peril`, `claim_status`, amounts, `region`, `severity` | Genie / structured tools, PII guardrails, fine-tune labels |
| `claim_notes` | one free-text adjuster note per claim, paired with its `severity` | RAG text, fine-tune training data |
| `doc_chunks` | chunked knowledge docs (CDF on) | Vector Search Delta-Sync index |
| `/Volumes/.../docs` | 8 policy wordings + claims-handling guidelines (`.md`) | Knowledge Assistant, RAG index |

All synthetic — no real customer data.

## Run it

```bash
databricks bundle run -t dev claims_demo_setup_job
```

Or run `00_setup_and_data.py` interactively on serverless.

## Used by

- `agent_bricks_claims/` — KA + Supervisor + custom RAG
- `ai_governance/` — AI Gateway + guardrails + MCP + observability
- `ai_runtime/` — Ray + LoRA fine-tune for claims triage

Config (catalog/schema/models) is in `config.py`.
