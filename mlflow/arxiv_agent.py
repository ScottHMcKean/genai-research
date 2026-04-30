"""LangGraph ReAct agent for the MLflow eval walkthrough.

The agent uses `create_react_agent` over `ChatDatabricks` with the two arXiv tools.
System prompt is loaded from the Unity Catalog Prompt Registry by alias, so v1 vs v2
can be swapped without changing code.

Tracing:
- `mlflow.langchain.autolog()` (enabled in the notebook) captures the graph and LLM.
- The arXiv tools in `arxiv_tools.py` produce their own named TOOL spans.
- `run_turn` tags each top-level trace with session + user metadata so multi-turn
  conversations group in the MLflow Sessions view.
"""

from __future__ import annotations

from typing import Any

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from arxiv_tools import ALL_TOOLS


def load_prompt_by_alias(prompt_fqn: str, alias: str) -> str:
    """Load a prompt template from the UC Prompt Registry by alias (e.g. 'production')."""
    return mlflow.genai.load_prompt(f"prompts:/{prompt_fqn}@{alias}").template


def build_agent(
    endpoint: str,
    system_prompt: str,
):
    """Build a LangGraph ReAct agent with the arXiv tools."""
    llm = ChatDatabricks(endpoint=endpoint)
    return create_react_agent(model=llm, tools=ALL_TOOLS, prompt=system_prompt)


def run_turn(
    agent: Any,
    question: str,
    session_id: str,
    user: str | None = None,
) -> str:
    """Invoke the agent for one turn and tag the trace with session metadata."""
    metadata = {"mlflow.trace.session": session_id}
    if user:
        metadata["mlflow.trace.user"] = user
    mlflow.update_current_trace(metadata=metadata)

    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content
