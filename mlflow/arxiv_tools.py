"""arXiv tools for the MLflow evaluation walkthrough.

Two functions hit the public arXiv API. Both are wrapped in two ways:

1. `@mlflow.trace(span_type=SpanType.TOOL, name=...)` on the underlying function so
   every invocation produces a named TOOL span that judges can read from the trace.
2. `@tool` from langchain_core.tools so the LangGraph ReAct agent can call them.

The arXiv API is public and requires no authentication. Be polite: one request at a
time is fine, but avoid tight loops.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from typing import Any

import feedparser
import mlflow
from langchain_core.tools import tool
from mlflow.entities import SpanType

ARXIV_API = "http://export.arxiv.org/api/query"


def _parse_entry(entry: Any) -> dict[str, Any]:
    arxiv_id = entry.id.rsplit("/", 1)[-1]
    # Strip version suffix like "2301.12345v2" -> "2301.12345"
    if "v" in arxiv_id and arxiv_id.rsplit("v", 1)[-1].isdigit():
        arxiv_id = arxiv_id.rsplit("v", 1)[0]
    return {
        "arxiv_id": arxiv_id,
        "title": entry.title.strip().replace("\n ", " "),
        "authors": [a.name for a in entry.authors],
        "published": entry.published,
        "summary": entry.summary.strip().replace("\n", " "),
        "categories": [t["term"] for t in getattr(entry, "tags", [])],
    }


@mlflow.trace(span_type=SpanType.TOOL, name="search_arxiv")
def search_arxiv(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search arXiv and return a list of paper summaries.

    Args:
        query: Free-text search query. Passed as `all:<query>` to the arXiv API.
        max_results: How many results to return (1-20).

    Returns:
        List of dicts with keys: arxiv_id, title, authors, published,
        summary (abstract), categories.
    """
    max_results = max(1, min(int(max_results), 20))
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=20) as r:
        feed = feedparser.parse(r.read())
    return [_parse_entry(e) for e in feed.entries]


@mlflow.trace(span_type=SpanType.TOOL, name="fetch_arxiv_paper")
def fetch_arxiv_paper(arxiv_id: str) -> dict[str, Any]:
    """Fetch the full metadata for a single arXiv paper by ID.

    Args:
        arxiv_id: e.g. "2301.12345" (version suffix optional).

    Returns:
        Dict with arxiv_id, title, authors, published, summary, categories.
        Returns {"error": "not found"} if the id is unknown.
    """
    params = {"id_list": arxiv_id}
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=20) as r:
        feed = feedparser.parse(r.read())
    if not feed.entries:
        return {"error": "not found", "arxiv_id": arxiv_id}
    return _parse_entry(feed.entries[0])


search_arxiv_tool = tool(search_arxiv)
fetch_arxiv_paper_tool = tool(fetch_arxiv_paper)

ALL_TOOLS = [search_arxiv_tool, fetch_arxiv_paper_tool]
