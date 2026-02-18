"""Perplexity API client."""

import asyncio
import logging
import time

from perplexity import Perplexity

from standard_benchmark.data.models import SearchResponse, Source
from standard_benchmark.search.base import BaseSearchClient

logger = logging.getLogger(__name__)


class PerplexityClient(BaseSearchClient):
    """Client for Perplexity AI API."""

    def __init__(self, api_key: str, base_url: str = "https://api.perplexity.ai"):
        """Initialize Perplexity client.

        Args:
            api_key: Perplexity API key
            base_url: Base URL for Perplexity API (not used with SDK)
        """
        super().__init__(api_key)
        self.client = Perplexity(api_key=api_key)

    async def search(
        self, query: str, model: str = "sonar", **kwargs
    ) -> SearchResponse:
        """Execute a search query asynchronously.

        Args:
            query: Search query string
            model: Model to use (default: 'sonar')
            **kwargs: Additional parameters

        Returns:
            SearchResponse with answer and sources
        """
        start_time = time.time()

        try:
            # Run the synchronous SDK call in a thread pool
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,  # type: ignore[arg-type]
                messages=[{"role": "user", "content": query}],
                model=model,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract answer from choices
            answer = ""
            if completion.choices and len(completion.choices) > 0:
                answer = str(completion.choices[0].message.content or "")

            # Parse sources from search_results
            sources = []
            if hasattr(completion, "search_results") and completion.search_results:
                for result in completion.search_results:
                    # Handle both dict and object result items
                    if isinstance(result, dict):
                        url = result.get("url", "")
                        title = result.get("title", "")
                        snippet = result.get("snippet", "")
                    else:
                        url = getattr(result, "url", "")
                        title = getattr(result, "title", "")
                        snippet = getattr(result, "snippet", "")

                    sources.append(
                        Source(
                            url=url,
                            title=title,
                            snippet=snippet,
                            domain=self._extract_domain(url),
                        )
                    )

            # Also add citations if available
            if hasattr(completion, "citations") and completion.citations:
                for citation_url in completion.citations:
                    # Only add if not already in sources
                    if not any(s.url == citation_url for s in sources):
                        sources.append(
                            Source(
                                url=citation_url,
                                domain=self._extract_domain(citation_url),
                            )
                        )

            # Get usage info
            usage = {}
            if hasattr(completion, "usage") and completion.usage is not None:
                usage = {
                    "prompt_tokens": (
                        completion.usage.prompt_tokens
                        if hasattr(completion.usage, "prompt_tokens")
                        else None
                    ),
                    "completion_tokens": (
                        completion.usage.completion_tokens
                        if hasattr(completion.usage, "completion_tokens")
                        else None
                    ),
                    "total_tokens": (
                        completion.usage.total_tokens
                        if hasattr(completion.usage, "total_tokens")
                        else None
                    ),
                }

            return SearchResponse(
                system=self.get_name(),
                response_text=answer,
                sources=sources,
                latency_ms=latency_ms,
                metadata={
                    "model": model,
                    "usage": usage,
                    "citations_count": (
                        len(completion.citations)
                        if hasattr(completion, "citations")
                        and completion.citations is not None
                        else 0
                    ),
                    "search_results_count": (
                        len(completion.search_results)
                        if hasattr(completion, "search_results")
                        and completion.search_results is not None
                        else 0
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Perplexity search failed: {e}")
            raise

    def get_name(self) -> str:
        """Get system name."""
        return "perplexity"

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse

        try:
            return urlparse(url).netloc
        except Exception:
            return ""
