# Migrating a notebook agent to this pattern

If you have a "deploy-everything" notebook (writes `agent.py` with `%%writefile`, builds the
RAG index, registers tools, logs/registers/vendors a model, and deploys — all in run-order
cells), here's the short path to this shape. It's intentionally small; keep it that way.

## What moves where

| In the notebook | Becomes |
|---|---|
| `%%writefile agent.py` cell | a real, versioned [`agent.py`](agent.py) (lint + unit-test it) |
| scattered `CATALOG`/`SCHEMA`/endpoint literals | one [`config.py`](config.py) imported everywhere |
| hand-wired `UCFunctionToolkit` + VS client | tools via **managed MCP servers** (`agent.py` lists them) |
| per-instance request state (`self._request_ctx`) | tag the trace per request in `predict` — hold nothing on `self` |
| raw char-count truncation of tool JSON | JSON-safe `cap_tool_output` |
| `conda_env` dict + `%pip install` spread across cells | one pruned [`requirements.txt`](requirements.txt) |
| `add_libraries_to_model` vendoring + `_max_ver` polling | drop it — deploy the exact registered version, or skip the endpoint entirely |
| `agents.deploy` to a Model Serving endpoint | **serve in-process on a Databricks App** ([`app/`](app/)) — cheaper, no cold-start hop |
| manual "run order matters" cells | a bundle job: [`00_setup.py`](00_setup.py) → [`01_evaluate.py`](01_evaluate.py) |
| no evaluation | an `mlflow.genai.evaluate` gate before you ship |

## Steps

1. **Lift `agent.py` out of the notebook** into a file; replace per-instance request state with
   per-request trace tagging, and wrap oversized tool output as valid JSON.
2. **Centralize config** in `config.py`; delete inline literals and any glob-based config loader.
3. **Expose tools via managed MCP** (Functions + Vector Search) instead of hand-wiring a toolkit.
4. **Build data/index/tools in `00_setup.py`** with the SDK — no in-notebook `%pip install` +
   `restartPython` (it can downgrade protobuf and break the serverless kernel).
5. **Serve on an App**: a thin FastAPI server imports `AGENT` and streams `predict_stream`.
6. **Wrap it in a folder-scoped bundle** (`databricks.yml`) with `dev`/`prod` targets, and add the
   `mlflow.genai.evaluate` gate in `01_evaluate.py`.

That's it — see each file for the concrete version.
