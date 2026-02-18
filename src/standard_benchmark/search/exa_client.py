"""Exa API client."""

import asyncio
import logging
import time

from exa_py import Exa

from standard_benchmark.data.models import SearchResponse, Source
from standard_benchmark.search.base import BaseSearchClient

logger = logging.getLogger(__name__)


class ExaClient(BaseSearchClient):
    """Client for Exa search API."""

    def __init__(self, api_key: str, base_url: str = "https://api.exa.ai"):
        """Initialize Exa client.

        Args:
            api_key: Exa API key
            base_url: Base URL for Exa API (not used with SDK)
        """
        super().__init__(api_key)
        self.client = Exa(api_key)

    async def search(self, query: str, text: bool = True, **kwargs) -> SearchResponse:
        """Execute a search query asynchronously.

        Args:
            query: Search query string
            text: Whether to include text content
            **kwargs: Additional parameters

        Returns:
            SearchResponse with answer and sources
        """
        start_time = time.time()

        try:
            # Run the synchronous SDK call in a thread pool
            result = await asyncio.to_thread(
                self.client.answer, query, text=text, **kwargs  # type: ignore[arg-type]
            )

            latency_ms = (time.time() - start_time) * 1000

            # Parse sources from citations
            sources = []
            if hasattr(result, "citations") and result.citations:
                for citation in result.citations:
                    # Handle both dict and object citation items
                    if isinstance(citation, dict):
                        url = citation.get("url", "")
                        title = citation.get("title", "")
                        snippet = citation.get("text", "")
                    else:
                        url = getattr(citation, "url", "")
                        title = getattr(citation, "title", "")
                        snippet = getattr(citation, "text", "")

                    sources.append(
                        Source(
                            url=url,
                            title=title,
                            snippet=snippet,
                            domain=self._extract_domain(url),
                        )
                    )

            # Get answer and cost
            answer = getattr(result, "answer", "")
            cost = getattr(result, "costDollars", {})

            return SearchResponse(
                system=self.get_name(),
                response_text=answer,
                sources=sources,
                latency_ms=latency_ms,
                metadata={"cost": cost, "citations_count": len(sources)},
            )

        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            raise

    def get_name(self) -> str:
        """Get system name."""
        return "exa"

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse

        try:
            return urlparse(url).netloc
        except Exception:
            return ""
