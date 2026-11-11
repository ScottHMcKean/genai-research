# Unified tracing and LLM cost: how cost reaches a span, what to enable, and how this overlay helps

This note explains how trace cost reaches a span. It tells you what to enable, why
cost can be wrong or absent, and how the overlay in this folder corrects it. The
note uses the MLflow and Databricks docs cited at the end. It also uses live
inspection of two workspaces: the `uaigw_fe` gateway traces and the
`shm_skunkworks_catalog.forecast_demo` MLflow application traces.

## 1. Two kinds of "unified tracing". Do not confuse them.

Both kinds write OpenTelemetry (OTel) spans into Unity Catalog tables. The table
names follow the pattern `<catalog>.<schema>.<prefix>_otel_spans` (and `_otel_logs`,
`_otel_metrics`, `_otel_annotations`). The two kinds are different features. They
record cost differently.

| | MLflow application tracing | AI Gateway unified tracing |
| --- | --- | --- |
| Traces | your instrumented app or agent | all Unity Gateway traffic in the metastore |
| Enabled by | a per-experiment `trace_location` | a one-time metastore-admin setup |
| Cost on the span | yes (`mlflow.llm.cost`), if the conditions are met | no cost attribute |
| Token usage | `mlflow.chat.tokenUsage` | `mlflow.chat.tokenUsage` and `gen_ai.usage.*` |
| Trace-level cost | `mlflow.trace.cost` in `_trace_metadata` | none (no `_trace_metadata` table) |

The customer workspace is the left column. Cost is present, but the rate is wrong.
The `uaigw_fe` test bed is the right column. Cost is absent, and only tokens exist.

## 2. How cost reaches a span (MLflow application tracing)

MLflow records token usage and an estimated USD cost on each LLM-call span. MLflow
then aggregates both to the trace.

- Token usage is the span attribute `mlflow.chat.tokenUsage`. It holds
  `input_tokens`, `output_tokens`, and `total_tokens`.
- Cost is the span attribute `mlflow.llm.cost`. It holds `input_cost`,
  `output_cost`, and `total_cost` in USD.
- Model and provider identity are `mlflow.llm.model` and `mlflow.llm.provider`.
- Trace totals are `mlflow.trace.cost` and `mlflow.trace.tokenUsage`. They are in
  the `<prefix>_trace_metadata` view.

MLflow does not price per DBU. MLflow reads a per-token list price from the LiteLLM
price table (`cost_per_token`, with a built-in fallback). MLflow keys the price on
the model name and provider, then multiplies by the token counts. The skunkworks
data confirms this. `databricks-claude-sonnet-4-5` gave $3 per 1M input tokens and
$15 per 1M output tokens. This is a per-token list price. No DBU is in the
calculation.

## 3. What to enable to get cost

For MLflow application tracing (to get `mlflow.llm.cost`):

1. Use the correct MLflow version. Token tracking needs 3.2.0 or later. Cost
   tracking needs 3.10.0 or later. The Unity Catalog trace location needs 3.14.0 or
   later.
2. Compute cost on the client. Databricks-managed MLflow does not compute cost on
   the server. Install LiteLLM in the application, or set the `mlflow.llm.cost`
   attribute on the span yourself.
3. Make the model name on the span match the price table. A custom serving-endpoint
   alias (for example `my-claude-endpoint`) does not identify a model. The cost is
   then $0.00, even when tokens are present.
4. Bind the experiment to a Unity Catalog trace location at creation time. Set
   `MLFLOW_TRACING_SQL_WAREHOUSE_ID`. Set `trace_location=UnityCatalog(catalog,
   schema, table_prefix)`. Create the schema first. Grant `USE CATALOG`,
   `USE SCHEMA`, `SELECT`, and `MODIFY` on the trace tables. `ALL_PRIVILEGES` alone
   is not enough. You cannot reassign the binding later.

For AI Gateway unified tracing:

1. An account admin enables the Enhanced Unity Gateway beta.
2. A metastore admin opens AI Gateway > Govern > Traces > Set up tracing.
3. The admin selects a catalog and schema, then saves.

This captures `gen_ai.*` and `databricks.*` attributes and token usage. It records
no cost attribute. Delivery is best-effort, so keep audit logs as the system of
record.

## 4. Why cost is wrong or absent. The two cases this overlay targets.

1. Wrong rate (the customer). Cost is present. The LiteLLM list price does not match
   the customer Azure North Europe rate. MLflow estimates at an AWS/GCP list rate
   (about $0.07 per DBU-equivalent) instead of Azure NE ($0.105 list, about $0.092
   effective). The number is self-consistent, but the rate card is wrong.
2. No cost (AI Gateway and uaigw). Gateway spans carry tokens but no
   `mlflow.llm.cost`. Nothing exists to correct, so the overlay must derive cost
   from tokens. MLflow shows the same $0.00 case when the model name is a custom
   endpoint alias.

## 5. How this overlay helps

The overlay is a read-only set of Unity Catalog views over the existing
`_otel_spans` table. It never writes to the MLflow-managed trace tables. For each
span, it detects the case and corrects the cost.

- Span has `mlflow.llm.cost` (the `dbu_multiplier` mode).
  `corrected = mlflow_total_cost × (target_rate / mlflow_assumed_rate)`. The target
  rate is the Azure list rate or the effective rate from the config table. The
  seeded rates give 1.5× (list) or 1.3143× (effective). The skunkworks test confirms
  this. The multiplier applies exactly. The sum of the original span cost equals
  MLflow's own `mlflow.trace.cost` for 100% of traces. The correction uses a number
  that reconciles to source.
- Span has tokens only (the `token` mode).
  `corrected = billable_input/1e6 × input_rate + cache_read/1e6 × cache_rate +
  output/1e6 × output_rate`. This prices from tokens directly. MLflow recommends
  this path when the model name does not resolve. It is the only path that works on
  gateway traces. The uaigw test confirms it.

The overlay also handles the parts that make a naive query wrong:

- Model identity. It resolves the model from `gen_ai.request.model`,
  `mlflow.llm.model`, the child span, or the endpoint name. It strips `system.ai.`
  and `databricks-` prefixes so the name matches the rate table.
- Double counting. A gateway call writes a parent `LLM` span and a child `chat`
  span with identical tokens. The trace view keeps only the top-most cost-bearing
  span in each branch (75,247 → 37,624 on uaigw). The MLflow application traces put
  cost on leaf spans only, so the overlay drops none there.
- Cache reads. `cache_read_input_tokens` is inside `input_tokens` and can be almost
  all of it (about 92% for one uaigw model). The overlay prices cache reads on a
  separate, lower rate.
- Unmatched models. The overlay passes them through, never drops them, and flags
  them `rate_matched = false`.

Rates live only in `trace_cost_rate_config`. No rate is in code. To change a rate,
edit a row. To switch between list and effective, change the `pricing_mode`
setting.

## 6. The exit. When to delete this overlay.

This overlay is interim. Delete it when MLflow trace cost prices from the platform
instead of the LiteLLM list table. That is, delete it when Gateway/FMAPI pricing (or
a `system`-table cost source) feeds the trace cost directly, so the recorded
`mlflow.llm.cost` already reflects the customer rate. Until then, keep the seeded
token rates current. Treat every corrected figure as an estimate. The teardown
steps are in `README.md`.

## Sources

- MLflow — Token usage and cost tracking:
  https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/index.html
- Databricks — MLflow traces in Unity Catalog:
  https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog
- Databricks — AI Gateway unified trace table (Beta):
  https://docs.databricks.com/aws/en/ai-gateway/unified-trace-table
