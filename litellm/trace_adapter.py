"""Adapter between Databricks AI Gateway inference tables and litellm (zerobus) traces.

Two trace shapes, one request-per-row each, that we want to query and reason about together:

* **AI Gateway inference table** (Mosaic AI Gateway payload logging): request_id, event_time,
  status_code, latency_ms, destination_model, and raw JSON `request` / `response` STRINGs.
* **litellm trace table** (what litellm's zerobus callback writes): id, model, status,
  start_time/end_time, prompt/completion/total tokens, response_cost, and VARIANT
  `messages` / `response`.  See litellm.integrations.zerobus.row for the full 40 columns.

This module is pure Python (no `spark`/`dbutils`), so it imports cleanly in a job, a notebook,
or a unit test. It provides:

  * `gateway_row_to_unified` / `litellm_row_to_unified` -- normalize either row into one dict.
  * `unified_to_otel_attributes` -- map the unified row onto OpenTelemetry GenAI semantic
    convention attributes (gen_ai.*), the common denominator both sources can satisfy.
  * `verify_otel_compatible` -- assert that rows from BOTH sources yield the same required
    gen_ai.* attribute set with the right types (the "are these two traces compatible?" check).
  * `unified_view_sql` -- the CREATE OR REPLACE VIEW that unions both tables in SQL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

# --- OpenTelemetry GenAI semantic-convention attribute keys ----------------------------------
# https://opentelemetry.io/docs/specs/semconv/gen-ai/
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# Attributes we require BOTH sources to populate for the traces to be considered compatible.
REQUIRED_OTEL_ATTRS: dict[str, type] = {
    GEN_AI_SYSTEM: str,
    GEN_AI_OPERATION: str,
    GEN_AI_REQUEST_MODEL: str,
    GEN_AI_USAGE_INPUT_TOKENS: int,
    GEN_AI_USAGE_OUTPUT_TOKENS: int,
}


@dataclass
class UnifiedTrace:
    """One LLM request, normalized across sources. Times are tz-aware UTC datetimes."""

    source: str                      # "gateway" | "litellm"
    request_id: str | None
    provider: str                    # gen_ai.system
    operation: str                   # gen_ai.operation.name (e.g. "chat")
    request_model: str | None
    response_model: str | None
    response_id: str | None
    status: str                      # "success" | "error"
    status_code: int | None
    start_time: datetime | None
    end_time: datetime | None
    duration_ms: int | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    requester: str | None
    messages: Any = None             # parsed request messages (list) when available
    response_text: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        return d


# --- helpers ---------------------------------------------------------------------------------
def _as_obj(value: Any) -> Any:
    """VARIANT/MAP columns arrive as dicts; STRING JSON columns as str. Normalize to objects."""
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _get(obj: Any, *path: Any, default: Any = None) -> Any:
    """Safe nested lookup across dicts/lists."""
    cur = obj
    for key in path:
        if isinstance(key, int) and isinstance(cur, (list, tuple)):
            cur = cur[key] if -len(cur) <= key < len(cur) else default
        elif isinstance(cur, Mapping):
            cur = cur.get(key, default)
        else:
            return default
        if cur is None:
            return default
    return cur


def _to_dt(value: Any) -> datetime | None:
    """Accept datetime, epoch seconds/millis, or ISO string -> tz-aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        # Heuristic: >1e12 looks like milliseconds.
        secs = value / 1000.0 if value > 1e12 else float(value)
        return datetime.fromtimestamp(secs, tz=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# --- normalizers -----------------------------------------------------------------------------
def gateway_row_to_unified(row: Mapping[str, Any]) -> UnifiedTrace:
    """Normalize a Mosaic AI Gateway inference-table row (dict-like) into a UnifiedTrace."""
    req = _as_obj(row.get("request"))
    resp = _as_obj(row.get("response"))
    usage = _get(resp, "usage", default={}) or {}

    status_code = _int(row.get("status_code"))
    start = _to_dt(row.get("event_time") or row.get("request_time"))
    duration_ms = _int(row.get("latency_ms"))
    end = None
    if start is not None and duration_ms is not None:
        end = datetime.fromtimestamp(start.timestamp() + duration_ms / 1000.0, tz=timezone.utc)

    return UnifiedTrace(
        source="gateway",
        request_id=row.get("request_id") or row.get("databricks_request_id"),
        provider=row.get("destination_type") or "databricks",
        operation="chat",
        request_model=row.get("destination_model") or _get(req, "model"),
        response_model=_get(resp, "model"),
        response_id=_get(resp, "id"),
        status="success" if status_code == 200 else "error",
        status_code=status_code,
        start_time=start,
        end_time=end,
        duration_ms=duration_ms,
        input_tokens=_int(usage.get("prompt_tokens")),
        output_tokens=_int(usage.get("completion_tokens")),
        total_tokens=_int(usage.get("total_tokens")),
        requester=row.get("requester"),
        messages=_get(req, "messages"),
        response_text=_get(resp, "choices", 0, "message", "content"),
        raw=dict(row),
    )


def litellm_row_to_unified(row: Mapping[str, Any]) -> UnifiedTrace:
    """Normalize a litellm/zerobus trace-table row (dict-like) into a UnifiedTrace."""
    resp = _as_obj(row.get("response"))
    status = (row.get("status") or "").lower()
    duration_ms = _int(row.get("response_time"))  # litellm logs response_time in ms
    call_type = (row.get("call_type") or "chat").lower()
    operation = "chat" if "completion" in call_type else call_type

    return UnifiedTrace(
        source="litellm",
        request_id=row.get("id") or row.get("litellm_call_id"),
        provider=row.get("custom_llm_provider") or "databricks",
        operation=operation,
        request_model=row.get("model"),
        response_model=_get(resp, "model") or row.get("model"),
        response_id=_get(resp, "id") or row.get("id"),
        status="success" if status in ("success", "") else "error",
        status_code=200 if status == "success" else None,
        start_time=_to_dt(row.get("start_time")),
        end_time=_to_dt(row.get("end_time")),
        duration_ms=duration_ms,
        input_tokens=_int(row.get("prompt_tokens")),
        output_tokens=_int(row.get("completion_tokens")),
        total_tokens=_int(row.get("total_tokens")),
        requester=row.get("end_user") or row.get("user_id"),
        messages=_as_obj(row.get("messages")),
        response_text=_get(resp, "choices", 0, "message", "content"),
        raw=dict(row),
    )


# --- OpenTelemetry mapping --------------------------------------------------------------------
def unified_to_otel_attributes(u: UnifiedTrace) -> dict[str, Any]:
    """Map a UnifiedTrace onto OTel GenAI semantic-convention attributes (drops None values)."""
    attrs: dict[str, Any] = {
        GEN_AI_SYSTEM: u.provider,
        GEN_AI_OPERATION: u.operation,
        GEN_AI_REQUEST_MODEL: u.request_model,
        GEN_AI_RESPONSE_MODEL: u.response_model,
        GEN_AI_RESPONSE_ID: u.response_id,
        GEN_AI_USAGE_INPUT_TOKENS: u.input_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: u.output_tokens,
    }
    return {k: v for k, v in attrs.items() if v is not None}


def otel_span_name(u: UnifiedTrace) -> str:
    """OTel GenAI span-name convention: '<operation> <model>'."""
    return f"{u.operation} {u.request_model}".strip()


def verify_otel_compatible(unified_rows: Sequence[UnifiedTrace]) -> dict[str, Any]:
    """Assert both sources emit the required gen_ai.* attributes with the right types.

    Returns a report dict. Raises AssertionError on the first incompatibility so it can gate a
    job. "Compatible" means: every required attribute is present and correctly typed for rows
    from EACH source, and both sources are actually represented.
    """
    sources = {u.source for u in unified_rows}
    report: dict[str, Any] = {"sources": sorted(sources), "checked": len(unified_rows), "per_source": {}}

    for src in sorted(sources):
        rows = [u for u in unified_rows if u.source == src]
        missing: dict[str, int] = {}
        mistyped: dict[str, int] = {}
        for u in rows:
            attrs = unified_to_otel_attributes(u)
            for key, typ in REQUIRED_OTEL_ATTRS.items():
                if key not in attrs:
                    missing[key] = missing.get(key, 0) + 1
                elif not isinstance(attrs[key], typ):
                    mistyped[key] = mistyped.get(key, 0) + 1
        report["per_source"][src] = {"rows": len(rows), "missing": missing, "mistyped": mistyped}
        assert not missing, f"[{src}] rows missing required OTel attrs: {missing}"
        assert not mistyped, f"[{src}] rows with mistyped OTel attrs: {mistyped}"

    assert {"gateway", "litellm"} <= sources, (
        f"expected both 'gateway' and 'litellm' traces to validate OTel compatibility, got {sorted(sources)}"
    )
    report["compatible"] = True
    return report


# --- SQL view --------------------------------------------------------------------------------
def unified_view_sql(view_name: str, litellm_table: str, gateway_table: str) -> str:
    """CREATE OR REPLACE VIEW unioning both trace tables into the unified schema.

    Column projections mirror the Python normalizers so the view and `*_row_to_unified` agree.
    Gateway `request`/`response` are JSON STRINGs (parsed with get_json_object); litellm
    `messages`/`response` are VARIANT (parsed with the `:` path operator).
    """
    return f"""CREATE OR REPLACE VIEW {view_name} AS
-- litellm (zerobus) traces
SELECT
  'litellm'                                              AS source,
  id                                                     AS request_id,
  COALESCE(custom_llm_provider, 'databricks')            AS provider,
  CASE WHEN lower(call_type) LIKE '%completion%' THEN 'chat' ELSE COALESCE(call_type, 'chat') END AS operation,
  model                                                  AS request_model,
  CASE WHEN lower(status) IN ('success', '') THEN 'success' ELSE 'error' END AS status,
  CASE WHEN lower(status) = 'success' THEN 200 ELSE NULL END AS status_code,
  start_time,
  end_time,
  CAST(response_time AS BIGINT)                          AS duration_ms,
  prompt_tokens                                          AS input_tokens,
  completion_tokens                                      AS output_tokens,
  total_tokens,
  COALESCE(end_user, user_id)                            AS requester,
  to_json(messages)                                      AS request_messages,
  CAST(response:choices[0].message.content AS STRING)    AS response_text
FROM {litellm_table}

UNION ALL

-- Databricks AI Gateway inference-table traces
SELECT
  'gateway'                                              AS source,
  request_id,
  COALESCE(destination_type, 'databricks')               AS provider,
  'chat'                                                 AS operation,
  COALESCE(destination_model, get_json_object(request, '$.model')) AS request_model,
  CASE WHEN status_code = 200 THEN 'success' ELSE 'error' END AS status,
  status_code,
  event_time                                             AS start_time,
  timestampadd(MILLISECOND, latency_ms, event_time)      AS end_time,
  latency_ms                                             AS duration_ms,
  CAST(get_json_object(response, '$.usage.prompt_tokens') AS BIGINT)     AS input_tokens,
  CAST(get_json_object(response, '$.usage.completion_tokens') AS BIGINT) AS output_tokens,
  CAST(get_json_object(response, '$.usage.total_tokens') AS BIGINT)      AS total_tokens,
  requester,
  get_json_object(request, '$.messages')                 AS request_messages,
  get_json_object(response, '$.choices[0].message.content') AS response_text
FROM {gateway_table}
"""
