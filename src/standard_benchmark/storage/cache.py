"""API response caching to avoid redundant calls."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from standard_benchmark.config.settings import settings
from standard_benchmark.data.models import SearchResponse
from standard_benchmark.utils.helpers import hash_text

logger = logging.getLogger(__name__)


class ResponseCache:
    """File-based cache for API responses."""

    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: int = 24):
        """Initialize the response cache.

        Args:
            cache_dir: Directory for cache files (defaults to settings.cache_dir)
            ttl_hours: Time-to-live for cached responses in hours
        """
        self.cache_dir = cache_dir or settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours

    def get(self, query: str, system: str) -> Optional[SearchResponse]:
        """Get cached response for a query.

        Args:
            query: Query string
            system: System name

        Returns:
            Cached SearchResponse or None if not found/expired
        """
        cache_key = self._get_cache_key(query, system)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            logger.debug(f"Cache miss for {system}: {query[:50]}...")
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            # Check TTL
            cached_time = datetime.fromisoformat(data["cached_at"])
            if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
                logger.debug(f"Cache expired for {system}: {query[:50]}...")
                cache_file.unlink()
                return None

            # Deserialize response
            response = SearchResponse(**data["response"])
            return response

        except Exception as e:
            logger.warning(f"Failed to load cache for {system}: {e}")
            return None

    def set(self, query: str, system: str, response: SearchResponse) -> None:
        """Cache a response.

        Args:
            query: Query string
            system: System name
            response: SearchResponse to cache
        """
        cache_key = self._get_cache_key(query, system)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            data = {
                "query": query,
                "system": system,
                "cached_at": datetime.now().isoformat(),
                "response": response.model_dump(),
            }

            with open(cache_file, "w") as f:
                json.dump(data, f, default=str)

            logger.debug(f"Cached response for {system}: {query[:50]}...")

        except Exception as e:
            logger.warning(f"Failed to cache response for {system}: {e}")

    def clear_expired(self) -> int:
        """Clear expired cache entries.

        Returns:
            Number of entries cleared
        """
        cleared = 0

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)

                cached_time = datetime.fromisoformat(data["cached_at"])
                if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
                    cache_file.unlink()
                    cleared += 1

            except Exception as e:
                logger.warning(f"Failed to check cache file {cache_file}: {e}")

        return cleared

    def clear_all(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        cleared = 0

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                cleared += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        return cleared

    def _get_cache_key(self, query: str, system: str) -> str:
        """Generate cache key for query and system.

        Args:
            query: Query string
            system: System name

        Returns:
            Cache key (hash)
        """
        combined = f"{system}:{query}"
        return hash_text(combined)
