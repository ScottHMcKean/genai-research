"""Pure-function tests for the scenario agent — run with no Databricks workspace.

    cd scenario_agent && python -m pytest tests/ -q

These cover the two fixes this reference implementation makes over the source notebook
(JSON-safe tool-output capping; no per-request state leakage) plus the message plumbing.
"""
import json
import sys
from pathlib import Path

# Import the agent/config modules from the folder root (tests/ is a subdir).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402
import config  # noqa: E402


def test_cap_tool_output_passthrough():
    assert agent.cap_tool_output('{"a":1}') == '{"a":1}'
    assert agent.cap_tool_output("", 10) == ""
    assert agent.cap_tool_output(None) == ""


def test_cap_tool_output_wraps_large_as_valid_json():
    big = "x" * 10_000
    out = agent.cap_tool_output(big, limit=100)
    parsed = json.loads(out)  # must still be valid JSON — the whole point of the fix
    assert parsed["truncated"] is True
    assert parsed["original_chars"] == 10_000
    assert len(parsed["content_prefix"]) == 100


def test_text_flattens_variants():
    assert agent._text(None) == ""
    assert agent._text("hi") == "hi"
    assert agent._text([{"text": "a"}, {"text": "b"}]) == "ab"
    assert agent._text([{"type": "output_text", "text": "x"}]) == "x"


def test_to_chat_function_call():
    msg = {"type": "function_call", "call_id": "c1", "name": "read_runs", "arguments": "{}"}
    out = agent._to_chat(msg)
    assert out[0]["role"] == "assistant"
    assert out[0]["tool_calls"][0]["id"] == "c1"
    assert out[0]["tool_calls"][0]["function"]["name"] == "read_runs"


def test_to_chat_function_call_output():
    msg = {"type": "function_call_output", "call_id": "c1", "output": "{}"}
    out = agent._to_chat(msg)
    assert out == [{"role": "tool", "content": "{}", "tool_call_id": "c1"}]


def test_no_per_request_state_on_instance():
    # The concurrency fix: request identity is never stored on the agent instance.
    a = agent.ScenarioAgent()
    assert not hasattr(a, "_request_ctx")


def test_config_fqns_are_three_level():
    for fqn in config.UC_TOOL_FUNCTIONS + [config.UC_MODEL, config.VS_INDEX,
                                           config.RUNS_TABLE, config.OUTPUT_TABLE]:
        assert fqn.count(".") == 2, fqn
