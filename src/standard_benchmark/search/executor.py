"""Async execution orchestrator for running all search systems."""

import asyncio
import logging
from typing import Optional

from standard_benchmark.config.settings import settings
from standard_benchmark.data.models import Query, SearchResponse
from standard_benchmark.search.base import BaseSearchClient
from standard_benchmark.search.exa_client import ExaClient
from standard_benchmark.search.linkup_client import LinkupClient
from standard_benchmark.search.perplexity_client import PerplexityClient
from standard_benchmark.search.tavily_client import TavilyClient
from standard_benchmark.storage.cache import ResponseCache

logger = logging.getLogger(__name__)


class SearchExecutor:
    """Orchestrate async search execution across multiple
    systems with concurrency control."""

    def __init__(
        self,
        linkup_api_key: Optional[str] = None,
        exa_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        perplexity_api_key: Optional[str] = None,
        use_cache: bool = True,
        max_concurrent_per_service: int = 5,
    ):
        """Initialize the search executor.

        Args:
            linkup_api_key: Linkup API key (defaults to settings)
            exa_api_key: Exa API key (defaults to settings)
            tavily_api_key: Tavily API key (defaults to settings)
            perplexity_api_key: Perplexity API key (defaults to settings)
            use_cache: Whether to use response caching
            max_concurrent_per_service: Max concurrent requests
                per service (default: 5)
        """
        self.clients: dict[str, BaseSearchClient] = {}
        self.semaphores: dict[str, asyncio.Semaphore] = {}

        # Initialize clients if API keys are provided
        if linkup_api_key or settings.linkup_api_key:
            self.clients["linkup"] = LinkupClient(
                api_key=linkup_api_key or settings.linkup_api_key
            )
            self.semaphores["linkup"] = asyncio.Semaphore(max_concurrent_per_service)

        if exa_api_key or settings.exa_api_key:
            self.clients["exa"] = ExaClient(
                api_key=exa_api_key or settings.exa_api_key
            )
            self.semaphores["exa"] = asyncio.Semaphore(max_concurrent_per_service)

        if tavily_api_key or settings.tavily_api_key:
            self.clients["tavily"] = TavilyClient(
                api_key=tavily_api_key or settings.tavily_api_key
            )
            self.semaphores["tavily"] = asyncio.Semaphore(max_concurrent_per_service)

        if perplexity_api_key or settings.perplexity_api_key:
            self.clients["perplexity"] = PerplexityClient(
                api_key=perplexity_api_key or settings.perplexity_api_key
            )
            self.semaphores["perplexity"] = asyncio.Semaphore(
                max_concurrent_per_service
            )

        self.max_concurrent_per_service = max_concurrent_per_service
        self.cache = ResponseCache() if use_cache else None

    async def execute_single_query(
        self, query: Query, systems: Optional[list[str]] = None, **kwargs
    ) -> dict[str, SearchResponse]:
        """Execute search for a single query across specified systems asynchronously.

        Args:
            query: Query object
            systems: List of system names (defaults to all available)
            **kwargs: Additional search parameters

        Returns:
            Dictionary mapping system name to SearchResponse
        """
        if systems is None:
            systems = list(self.clients.keys())

        # Filter to available systems
        systems = [s for s in systems if s in self.clients]

        if not systems:
            logger.warning("No valid search systems specified")
            return {}

        async def search_with_system(
            system_name: str,
        ) -> tuple[str, SearchResponse]:
            """Execute search for a single system with concurrency control."""
            async with self.semaphores[system_name]:
                try:
                    # Check cache first
                    if self.cache:
                        cached = self.cache.get(query.text, system_name)
                        if cached:
                            cached.query_id = query.id
                            return system_name, cached

                    # Execute search
                    client = self.clients[system_name]
                    response = await client.search_query_object(query, **kwargs)

                    # Cache response
                    if self.cache:
                        self.cache.set(query.text, system_name, response)

                    return system_name, response

                except Exception as e:
                    logger.error(
                        f"{system_name} search failed for query {query.id}: {e}"
                    )
                    return system_name, self._create_error_response(
                        system_name, query, str(e)
                    )

        # Execute searches concurrently
        tasks = [search_with_system(system) for system in systems]
        results_list = await asyncio.gather(*tasks)

        # Convert list of tuples to dict
        results = dict(results_list)
        return results

    async def execute_batch(
        self,
        queries: list[Query],
        systems: Optional[list[str]] = None,
        **kwargs,
    ) -> list[dict[str, SearchResponse]]:
        """Execute searches for multiple queries concurrently with rate limiting.

        Args:
            queries: List of Query objects
            systems: List of system names (defaults to all available)
            **kwargs: Additional search parameters

        Returns:
            List of dictionaries mapping system name to SearchResponse
        """
        # Execute all queries concurrently
        # Semaphores handle rate limiting per service
        tasks = [
            self.execute_single_query(query, systems, **kwargs) for query in queries
        ]
        results = await asyncio.gather(*tasks)

        return results

    def get_available_systems(self) -> list[str]:
        """Get list of available search systems.

        Returns:
            List of system names
        """
        return list(self.clients.keys())

    def _create_error_response(
        self, system: str, query: Query, error: str
    ) -> SearchResponse:
        """Create an error response.

        Args:
            system: System name
            query: Query object
            error: Error message

        Returns:
            SearchResponse with error metadata
        """
        return SearchResponse(
            query_id=query.id,
            system=system,
            response_text="",
            sources=[],
            latency_ms=0,
            metadata={"error": error, "status": "failed"},
        )
