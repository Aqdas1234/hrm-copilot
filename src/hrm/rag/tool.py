"""The policy search tool for agents (Part B).

make_policy_search_tool(service) returns a LangChain tool that agents call to look up HR
policies. It returns numbered excerpts with their document and section, so the agent can
quote and cite them. It searches only; writing the final answer is the agent's job.
"""

from typing import Protocol

from langchain_core.tools import BaseTool, tool

from hrm.rag.models import Category, RetrievedChunk

TOOL_NAME = "search_hr_policies"
NO_RESULTS = "No relevant policy sections found. The policies may not cover this."


class PolicySearch(Protocol):
    def search(
        self, query: str, k: int | None = None, *, category: Category | None = None
    ) -> list[RetrievedChunk]: ...


def make_policy_search_tool(service: PolicySearch, *, k: int = 5) -> BaseTool:
    @tool(TOOL_NAME)
    def search_hr_policies(query: str, category: Category | None = None) -> str:
        """Search Sourcegraph's HR policies (time off, parental leave, benefits, expenses,
        code of conduct) and return the most relevant excerpts with their sources.

        Args:
            query: The policy question, in plain words.
            category: Optional filter: benefits, conduct, expenses, parental_leave, time_off.
        """
        results = service.search(query, k, category=category)
        if not results:
            return NO_RESULTS
        return "\n\n".join(_format(n, r) for n, r in enumerate(results, start=1))

    return search_hr_policies


def _format(number: int, result: RetrievedChunk) -> str:
    chunk = result.chunk
    audience = " (US-based teammates only)" if chunk.audience == "us" else ""
    return (
        f"[{number}] {chunk.doc_id} > {' > '.join(chunk.section_path[1:]) or 'Introduction'}"
        f"{audience}\n{chunk.text}"
    )
