"""Linkup API client."""

import asyncio
import logging
import time

from linkup import LinkupClient as LinkupSDK

from standard_benchmark.data.models import SearchResponse, Source
from standard_benchmark.search.base import BaseSearchClient

logger = logging.getLogger(__name__)


class LinkupClient(BaseSearchClient):
    """Client for Linkup search API."""

    def __init__(self, api_key: str, base_url: str = "https://api.linkup.so/v1"):
        """Initialize Linkup client.

        Args:
            api_key: Linkup API key
            base_url: Base URL for Linkup API (not used with SDK)
        """
        super().__init__(api_key)
        self.client = LinkupSDK(api_key=api_key)

    async def search(
        self,
        query: str,
        depth: str = "standard",
        output_type: str = "sourcedAnswer",
        **kwargs,
    ) -> SearchResponse:
        """Execute a search query asynchronously.

        Args:
            query: Search query string
            depth: Search depth ('standard' or 'deep')
            output_type: Output type ('sourcedAnswer', 'searchResults', etc.)
            **kwargs: Additional parameters

        Returns:
            SearchResponse with answer and sources
        """
        start_time = time.time()

        try:
            # Run the synchronous SDK call in a thread pool
            response = await asyncio.to_thread(
                self.client.search,
                query=query,
                depth=depth,  # type: ignore[arg-type]
                output_type=output_type,  # type: ignore[arg-type]
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Parse sources - handle both dict and object responses
            sources = []
            response_sources = getattr(response, "sources", None) or (
                response.get("sources") if isinstance(response, dict) else None
            )
            if response_sources:
                for source in response_sources:
                    # Handle both dict and object source items
                    if isinstance(source, dict):
                        url = source.get("url", "")
                        title = source.get("name", "")
                        snippet = source.get("snippet", "")
                    else:
                        url = getattr(source, "url", "")
                        title = getattr(source, "name", "")
                        snippet = getattr(source, "snippet", "")

                    sources.append(
                        Source(
                            url=url,
                            title=title,
                            snippet=snippet,
                            domain=self._extract_domain(url),
                        )
                    )

            # Get answer - handle both dict and object responses
            answer = getattr(response, "answer", None) or (
                response.get("answer", "") if isinstance(response, dict) else ""
            )

            return SearchResponse(
                system=self.get_name(),
                response_text=answer,
                sources=sources,
                latency_ms=latency_ms,
                metadata={"depth": depth, "output_type": output_type},
            )

        except Exception as e:
            logger.error(f"Linkup search failed: {e}")
            raise

    def get_name(self) -> str:
        """Get system name."""
        return "linkup"

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse

        try:
            return urlparse(url).netloc
        except Exception:
            return ""
