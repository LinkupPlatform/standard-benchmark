"""Tavily API client."""

import asyncio
import logging
import time

from tavily import TavilyClient as TavilySDK

from standard_benchmark.data.models import SearchResponse, Source
from standard_benchmark.search.base import BaseSearchClient

logger = logging.getLogger(__name__)


class TavilyClient(BaseSearchClient):
    """Client for Tavily search API."""

    def __init__(self, api_key: str, base_url: str = "https://api.tavily.com"):
        """Initialize Tavily client.

        Args:
            api_key: Tavily API key
            base_url: Base URL for Tavily API (not used with SDK)
        """
        super().__init__(api_key)
        self.client = TavilySDK(api_key=api_key)

    async def search(
        self,
        query: str,
        search_depth: str = "basic",
        include_answer: bool = True,
        max_results: int = 10,
        **kwargs,
    ) -> SearchResponse:
        """Execute a search query asynchronously.

        Args:
            query: Search query string
            search_depth: Search depth ('basic' or 'advanced')
            include_answer: Whether to include AI-generated answer
            **kwargs: Additional parameters

        Returns:
            SearchResponse with answer and sources
        """
        start_time = time.time()

        try:
            # Run the synchronous SDK call in a thread pool
            response = await asyncio.to_thread(
                self.client.search,
                query,
                search_depth=search_depth,
                include_answer=include_answer,
                max_results=max_results,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Parse sources from results - handle both dict and object responses
            sources = []
            response_results = (
                response.get("results")
                if isinstance(response, dict)
                else getattr(response, "results", None)
            )
            if response_results:
                for result in response_results:
                    # Handle both dict and object result items
                    if isinstance(result, dict):
                        url = result.get("url", "")
                        title = result.get("title", "")
                        snippet = result.get("content", "")
                    else:
                        url = getattr(result, "url", "")
                        title = getattr(result, "title", "")
                        snippet = getattr(result, "content", "")

                    sources.append(
                        Source(
                            url=url,
                            title=title,
                            snippet=snippet,
                            domain=self._extract_domain(url),
                        )
                    )

            # Get response fields - handle both dict and object, and None values
            if isinstance(response, dict):
                answer = response.get("answer") or ""
                response_time = response.get("response_time", "")
                request_id = response.get("request_id", "")
                images = response.get("images", [])
            else:
                answer = getattr(response, "answer", None) or ""
                response_time = getattr(response, "response_time", "")
                request_id = getattr(response, "request_id", "")
                images = getattr(response, "images", [])

            return SearchResponse(
                system=self.get_name(),
                response_text=answer,
                sources=sources,
                latency_ms=latency_ms,
                metadata={
                    "search_depth": search_depth,
                    "response_time": response_time,
                    "request_id": request_id,
                    "images": images,
                },
            )

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            raise

    def get_name(self) -> str:
        """Get system name."""
        return "tavily"

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse

        try:
            return urlparse(url).netloc
        except Exception:
            return ""
