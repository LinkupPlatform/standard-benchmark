"""Cache for evaluation results to avoid redundant LLM API calls."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from standard_benchmark.data.models import EvaluationResult

logger = logging.getLogger(__name__)


class EvaluationCache:
    """File-based cache for evaluation results."""

    def __init__(
        self, cache_dir: Path = Path("./data/cache/evaluations"), ttl_hours: int = 168
    ):  # 7 days default
        """Initialize evaluation cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours (default: 168 = 7 days)
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _get_cache_key(
        self,
        query_text: str,
        response_text: str,
        system: str,
        config: dict,
        metric_name: Optional[str] = None,
    ) -> str:
        """Generate cache key from query, response, system, config, and optional metric.

        Args:
            query_text: Query text
            response_text: Response text
            system: System name
            config: Evaluation configuration (model, phases enabled, etc.)
            metric_name: Optional specific metric name for granular caching

        Returns:
            Cache key (hash)
        """
        # Include all factors that affect evaluation
        key_data = {
            "query": query_text,
            "response": response_text[:1000],  # First 1000 chars
            "system": system,
            "config": config,
        }
        if metric_name:
            key_data["metric"] = metric_name
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path.

        Args:
            cache_key: Cache key

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.json"

    def get(
        self, query_text: str, response_text: str, system: str, config: dict
    ) -> Optional[EvaluationResult]:
        """Get cached evaluation result (legacy method for full result).

        Args:
            query_text: Query text
            response_text: Response text
            system: System name
            config: Evaluation configuration

        Returns:
            Cached EvaluationResult or None
        """
        cache_key = self._get_cache_key(query_text, response_text, system, config)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)

            # Check TTL
            cached_at = datetime.fromisoformat(cache_data["cached_at"])
            if datetime.now() - cached_at > self.ttl:
                cache_path.unlink()  # Delete expired cache
                return None

            # Return cached result
            result = EvaluationResult(**cache_data["result"])
            return result

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def get_metric(
        self,
        query_text: str,
        response_text: str,
        system: str,
        config: dict,
        metric_name: str,
    ) -> Optional[float]:
        """Get cached metric score.

        Args:
            query_text: Query text
            response_text: Response text
            system: System name
            config: Evaluation configuration
            metric_name: Name of the metric (e.g., 'faithfulness', 'relevance')

        Returns:
            Cached metric score or None
        """
        cache_key = self._get_cache_key(
            query_text, response_text, system, config, metric_name
        )
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)

            # Check TTL
            cached_at = datetime.fromisoformat(cache_data["cached_at"])
            if datetime.now() - cached_at > self.ttl:
                cache_path.unlink()  # Delete expired cache
                return None

            # Return cached score
            score: float = float(cache_data["score"])
            return score

        except Exception as e:
            logger.warning(f"Failed to load metric cache: {e}")
            return None

    def set(
        self,
        query_text: str,
        response_text: str,
        system: str,
        config: dict,
        result: EvaluationResult,
    ):
        """Cache evaluation result (legacy method for full result).

        Args:
            query_text: Query text
            response_text: Response text
            system: System name
            config: Evaluation configuration
            result: EvaluationResult to cache
        """
        cache_key = self._get_cache_key(query_text, response_text, system, config)
        cache_path = self._get_cache_path(cache_key)

        try:
            cache_data = {
                "cached_at": datetime.now().isoformat(),
                "query": query_text[:100],  # First 100 chars for debugging
                "system": system,
                "result": result.model_dump(),
            }

            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    def set_metric(
        self,
        query_text: str,
        response_text: str,
        system: str,
        config: dict,
        metric_name: str,
        score: float,
    ):
        """Cache individual metric score.

        Args:
            query_text: Query text
            response_text: Response text
            system: System name
            config: Evaluation configuration
            metric_name: Name of the metric (e.g., 'faithfulness', 'relevance')
            score: Metric score to cache
        """
        cache_key = self._get_cache_key(
            query_text, response_text, system, config, metric_name
        )
        cache_path = self._get_cache_path(cache_key)

        try:
            cache_data = {
                "cached_at": datetime.now().isoformat(),
                "query": query_text[:100],  # First 100 chars for debugging
                "system": system,
                "metric": metric_name,
                "score": score,
            }

            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to cache metric: {e}")

    def get_metrics(
        self,
        query_text: str,
        response_text: str,
        system: str,
        config: dict,
        metric_names: list[str],
    ) -> dict[str, Optional[float]]:
        """Get multiple cached metric scores at once.

        Args:
            query_text: Query text
            response_text: Response text
            system: System name
            config: Evaluation configuration
            metric_names: List of metric names to retrieve

        Returns:
            Dictionary mapping metric names to scores (None if not cached)
        """
        results = {}
        for metric_name in metric_names:
            results[metric_name] = self.get_metric(
                query_text, response_text, system, config, metric_name
            )
        return results

    def set_metrics(
        self,
        query_text: str,
        response_text: str,
        system: str,
        config: dict,
        scores: dict[str, float],
    ):
        """Cache multiple metric scores at once.

        Args:
            query_text: Query text
            response_text: Response text
            system: System name
            config: Evaluation configuration
            scores: Dictionary mapping metric names to scores
        """
        for metric_name, score in scores.items():
            self.set_metric(
                query_text, response_text, system, config, metric_name, score
            )

    def clear_expired(self):
        """Remove expired cache entries."""
        removed = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                cached_at = datetime.fromisoformat(cache_data["cached_at"])
                if datetime.now() - cached_at > self.ttl:
                    cache_file.unlink()
                    removed += 1
            except Exception:
                pass

    def clear_all(self):
        """Remove all cache entries."""
        removed = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            removed += 1

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache stats
        """
        total = 0
        expired = 0
        total_size = 0

        for cache_file in self.cache_dir.glob("*.json"):
            total += 1
            total_size += cache_file.stat().st_size

            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                cached_at = datetime.fromisoformat(cache_data["cached_at"])
                if datetime.now() - cached_at > self.ttl:
                    expired += 1
            except Exception:
                pass

        return {
            "total_entries": total,
            "expired_entries": expired,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }
