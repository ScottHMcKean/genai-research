# FINS data — the common dataset

**The single script** (`generate_data.py`) that builds the one Unity Catalog schema every
use-case demo reads from. Run it once, then run any use case. Idempotent — safe to re-run.

## Run it

Open `generate_data.py` and **Run all** on serverless (no cluster needed). Or, optionally,
as a job: `databricks bundle run -t dev fins_data_setup_job`.

## What it creates (`shm_skunkworks_catalog.claims_demo`)

| Object | Description | Used by |
|--------|-------------|---------|
| `claims` | ~2,000 synthetic P&C claims — incl. **synthetic PII** (`customer_name`, `customer_email`, `customer_ssn`), `claim_type`, `peril`, `claim_status`, amounts, `region`, `severity` | governance (PII), ai_runtime (labels), structured tools |
| `claim_notes` | one free-text adjuster note per claim, paired with its `severity` | agents (RAG), ai_runtime (fine-tune) |
| `doc_chunks` | chunked knowledge docs (CDF on) | agents (Vector Search index) |
| `/Volumes/.../docs` | 8 policy wordings + claims-handling guidelines (`.md`) | agents (Knowledge Assistant + RAG) |
| `/Volumes/.../raw_pdfs` | real public ACORD / CMS insurance PDFs | document_intelligence |

All synthetic except the public PDFs — no real customer data.

## Adapt to your own discipline

This dataset only exists to illustrate the platform features. To reuse the demos with your
own data:

1. Point `CATALOG` / `SCHEMA` in `config.py` at your own Unity Catalog location.
2. Swap the generators (and `DOCS`) in `generate_data.py` for your domain, keeping the same
   table/column shape — or just register your existing tables under those names.

Each use-case folder has its own `config.py` mirroring these values, so a customer can take
a single folder and run it against their data.
