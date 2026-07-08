"""Custom RAG agent for insurance claims — MLflow ResponsesAgent.

Retrieve-then-generate over the claims knowledge docs (policy wordings +
handling guidelines) indexed in Databricks Vector Search, answered by a
foundation model. Logged and deployed as a Databricks Model Serving endpoint.
"""
import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from typing import Generator

# Kept in sync with config.py (log_model can't import the notebook config at serve time)
LLM_ENDPOINT = "databricks-claude-sonnet-4-5"
EMBED_ENDPOINT = "databricks-gte-large-en"
VS_ENDPOINT = "claims_vs"
VS_INDEX = "shm_skunkworks_catalog.claims_demo.doc_chunks_index"

SYSTEM_PROMPT = (
    "You are a claims assistant for a property & casualty insurer. Answer strictly "
    "from the retrieved policy wordings and claims-handling guidelines provided as "
    "context. Always cite the source document name in square brackets. If the answer "
    "is not in the context, say you don't have that information."
)


def _text(content) -> str:
    """Responses input content can be a str or a list of parts; normalize to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


class ClaimsRAGAgent(ResponsesAgent):
    def __init__(self):
        from databricks_langchain import ChatDatabricks
        from databricks.vector_search.client import VectorSearchClient

        self.llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1)
        self.index = VectorSearchClient(disable_notice=True).get_index(
            endpoint_name=VS_ENDPOINT, index_name=VS_INDEX
        )

    def _retrieve(self, query: str, k: int = 4):
        res = self.index.similarity_search(
            query_text=query,
            columns=["doc", "title", "chunk"],
            num_results=k,
        )
        return res.get("result", {}).get("data_array", []) or []

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        user_msgs = [m for m in request.input if m.role == "user"]
        query = _text(user_msgs[-1].content) if user_msgs else ""

        rows = self._retrieve(query)
        context = "\n\n".join(f"[{r[0]}] {r[2]}" for r in rows) or "(no documents found)"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        resp = self.llm.invoke(messages)
        text = resp.content if hasattr(resp, "content") else str(resp)
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=text, id="msg_1")]
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        result = self.predict(request)
        for item in result.output:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done", item=item
            )


mlflow.langchain.autolog()
AGENT = ClaimsRAGAgent()
mlflow.models.set_model(AGENT)
