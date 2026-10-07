# LiteLLM + Databricks Zerobus

Tests the [`litellm`](https://github.com/BerriAI/litellm) package against Databricks Foundation
Model APIs, exercises litellm's **Zerobus** logging callback (request traces → a Unity Catalog
Delta table), and provides an **adapter** that unifies Databricks **AI Gateway inference tables**
with litellm traces — verified **OpenTelemetry GenAI**-compatible.

Based on the litellm Zerobus guide: <https://docs.litellm.ai/docs/observability/zerobus>.
Runs on **shm-skunkworks** (AWS `us-west-2`, serverless-only).

## How litellm → Zerobus works

litellm's zerobus callback batches each request and POSTs JSON to the Zerobus **REST API**
(`{server_endpoint}/zerobus/v1/tables/<table>/insert`), authenticating as an OAuth **service
principal**. Because it's plain HTTPS (not the gRPC Zerobus SDK), it works on serverless compute.
One row per request; 40 columns (scalars + VARIANT) defined by
`litellm.integrations.zerobus.row`.

## Files

| File | What it does |
|------|--------------|
| `config.py` | Workspace, Zerobus endpoint, UC names, SP + secret-scope config. **Edit `GATEWAY_INFERENCE_TABLE`** to point at your real AI Gateway table. |
| `00_setup.py` | Creates the schema + trace table (DDL from litellm itself), the Zerobus service principal, grants (`MODIFY`+`SELECT`), and stores the OAuth secret. |
| `01_litellm_sdk_test.py` | litellm SDK smoke test vs Databricks FMAPI: completion, streaming, usage/cost, embeddings. |
| `02_zerobus_logging.py` | Registers `ZerobusLogger`, fires calls, flushes, and verifies rows land in `litellm_traces`. |
| `03_gateway_litellm_adapter.py` | Builds the unified SQL view + runs the Python normalizer + the OTel compatibility check. |
| `04_mlflow_traces.py` | Backfills the litellm + gateway rows into an **MLflow experiment** (UC trace storage) so they show in the MLflow Traces UI, with original timestamps. |
| `trace_adapter.py` | Pure-Python adapter: `gateway_row_to_unified`, `litellm_row_to_unified`, `unified_to_otel_attributes`, `verify_otel_compatible`, `unified_view_sql`. |

## The adapter

Both an AI Gateway inference row and a litellm trace row describe one LLM request. The adapter
maps each onto a common `UnifiedTrace` and onto OTel GenAI attributes
(`gen_ai.request.model`, `gen_ai.usage.input_tokens`, …):

- **SQL** — `unified_view_sql(...)` creates `unified_traces` as a `UNION ALL`, parsing the
  gateway's JSON `request`/`response` (`get_json_object`) and litellm's VARIANT columns (`:`).
- **Python** — the `*_row_to_unified` functions do the identical projection in-process.
- **OTel check** — `verify_otel_compatible(...)` asserts both sources yield the required
  `gen_ai.*` attributes with correct types; `03` then builds real OTel spans via the SDK to prove
  they're exportable.

> If you have no live AI Gateway, `03` creates a small sample `gateway_inference_sample` table in
> the Mosaic AI Gateway schema so the adapter + OTel check run end-to-end.

## Run it

Open the folder and run the notebooks in order (`00` → `04`) on serverless.

To run the full chain headless, sync the folder and `databricks jobs submit` a multi-task run
(`setup → sdk_test → zerobus_logging → adapter → mlflow_traces`):

```bash
databricks sync . /Workspace/Users/<you>/litellm --profile <profile>
# then submit a run whose notebook_task paths point at /Workspace/Users/<you>/litellm/00_setup, etc.
databricks jobs submit --profile <profile> --json '{"run_name":"litellm","tasks":[ ... ]}'
```

## Gotchas

- **litellm version** — the zerobus integration isn't in any *stable* release yet (stable 1.104.0
  predates the merge); it first ships in `v1.105.0-rc.1`, which `config.LITELLM_PIP` pins. Serverless
  also ships a `PIP_CONSTRAINT` file that caps litellm, so the notebooks clear it before installing.
  Bump the pin to the 1.105.0 stable once it's out.
- **`end_user` is NULL** — in SDK mode litellm does not map the `user=` arg to the row's `end_user`
  (that's a proxy-mode field). The `user=` value and `metadata={"tags": [...]}` surface as
  `request_tags` instead; `02` verifies on the trace `id` (== the litellm response id).
- **Async event loop** — Databricks notebooks already run an asyncio loop, so `02` runs its
  `acompletion` + `flush_queue()` on a dedicated thread with its own loop (not `asyncio.run()`).
- **Error 4024 / `authorization_details`** on ingest → the SP is missing explicit **table-level**
  `MODIFY`+`SELECT` (schema-level inherited grants aren't enough). `00_setup` grants these.
- **Region** — the Zerobus `server_endpoint` is region-specific (`<workspace-id>.zerobus.us-west-2…`).
  Wrong region ⇒ connection failures.
- **At-least-once** — Zerobus can duplicate rows on retry; dedupe on `id` if needed.
- **Cost estimate** — litellm's `completion_cost` may show `$0` for Databricks models it has no
  price entry for; that's not a failure. Databricks' own `response_cost` is logged on the row.
