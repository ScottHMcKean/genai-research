# Document Intelligence — parse → extract → evaluate (insurance docs)

One coherent flow for **intelligent document processing** on Databricks, on the shared
FINS data: parse real public insurance PDFs, extract structured fields, and score the
extraction against a hand-labelled golden set with MLflow.

```
raw insurance PDFs (Volume)
   │  ai_parse_document          (layout-aware text + OCR)
   ▼
parsed_docs ─► parsed_text
   │  ai_extract  (v2.1: per-field citations + confidence)
   ▼
extracted_fields ─► extracted_flat        (one column per field + confidence)
   │  mlflow.genai.evaluate      (table-driven: inputs+outputs+expectations → judges)
   ▼
MLflow: field accuracy, coverage recall, LLM judge, traces
```

## Run it (notebook-first)

Run on **serverless** (env 5), in order:

| # | Notebook | What it does |
|---|----------|--------------|
| 00 | `00_setup_and_data.ipynb` | Verifies the insurance PDFs are staged (by `fins_data/generate_data.py`) and writes the hand-labelled `golden_labels` table. |
| 01 | `01_parse_and_cost.ipynb` | `ai_parse_document` → `parsed_docs` / `parsed_text`; queries `AI_FUNCTIONS` cost + throughput. |
| 02 | `02_extract_fields.ipynb` | `ai_extract` (citations + confidence) → `extracted_fields` / `extracted_flat`; confidence-based review queue. |
| 03 | `03_evaluate_mlflow.ipynb` | Table-driven `mlflow.genai.evaluate` with tolerant code scorers + a deploy-client LLM judge. |

**Prerequisite:** run `fins_data/generate_data.py` once — it stages the real public
insurance PDFs into `raw_pdfs` in the common schema. All config (catalog/schema) is in
`config.py`; point it at your own catalog/schema to run on your documents.

## Optional: deploy from root as a job

The suite ships a Declarative Automation Bundle. To run this flow as a job instead of
interactively:

```bash
databricks bundle deploy -t dev
databricks bundle run   -t dev document_intelligence_job
```

## Data & labels

Ten genuine public documents (ACORD certificates + a CMS-1500) with **hand labels** —
messy on purpose (OCR, multi-column tables, template placeholders), which makes the eval
credible. Fields: `document_type`, `insurer_name`, `insured_name`, `policy_number`,
`effective_date`, `expiration_date`, `coverage_types`. The LLM judge scoring *lower* than
a naive string match is the point — it correctly rejects template placeholders.

## Alternate approaches

`alternates/` holds exploratory / alternative extraction paths kept for reference (direct
FMAPI `DocumentContent` with Ray, Docling, LayoutLMv3, layoutparser, few-shot multimodal
classification, YOLO). They are **not** part of the canonical flow above.
