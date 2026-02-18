"""Abstract base class for search clients."""

from abc import ABC, abstractmethod

from standard_benchmark.data.models import SearchResponse, Query


class BaseSearchClient(ABC):
    """Abstract base class for all search API clients."""

    def __init__(self, api_key: str, **kwargs):
        """Initialize the search client.

        Args:
            api_key: API key for authentication
            **kwargs: Additional client-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def search(self, query: str, **kwargs) -> SearchResponse:
        """Execute a search query asynchronously.

        Args:
            query: Search query string
            **kwargs: Additional search parameters

        Returns:
            SearchResponse with results
        """
        pass

    async def batch_search(self, queries: list[str], **kwargs) -> list[SearchResponse]:
        """Execute multiple search queries asynchronously.

        Args:
            queries: List of query strings
            **kwargs: Additional search parameters

        Returns:
            List of SearchResponse objects
        """
        import asyncio

        return await asyncio.gather(*[self.search(q, **kwargs) for q in queries])

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the search system.

        Returns:
            System name (e.g., 'linkup', 'exa', 'tavily', 'perplexity')
        """
        pass

    async def search_query_object(self, query: Query, **kwargs) -> SearchResponse:
        """Execute search for a Query object asynchronously.

        Args:
            query: Query object
            **kwargs: Additional search parameters

        Returns:
            SearchResponse with query_id populated
        """
        response = await self.search(query.text, **kwargs)
        response.query_id = query.id
        response.system = self.get_name()
        return response
