"""JSON file-based storage for all data."""

import logging
from pathlib import Path
from typing import Union

from standard_benchmark.data.models import Query, SearchResponse, EvaluationResult
from standard_benchmark.utils.helpers import (
    load_json,
    save_json,
    load_jsonl,
    save_jsonl,
    append_jsonl,
)

logger = logging.getLogger(__name__)


class JSONStore:
    """JSON-based storage for queries, responses, and evaluations."""

    def __init__(self, base_dir: Path = Path("./data")):
        """Initialize the JSON store.

        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_queries(
        self,
        queries: list[Query],
        path: Path | str,
        use_jsonl: bool = True,
    ) -> None:
        """Save queries to file.

        Args:
            queries: List of Query objects
            path: File path (relative to base_dir or absolute)
            use_jsonl: Whether to use JSON Lines format
        """
        path = self._resolve_path(path)

        data = [q.model_dump() for q in queries]

        if use_jsonl:
            save_jsonl(data, path)
        else:
            save_json(data, path)

        logger.info(f"Saved {len(queries)} queries to {path}")

    def load_queries(
        self, path: Union[str, Path], use_jsonl: bool = False
    ) -> list[Query]:
        """Load queries from file.

        Args:
            path: File path (relative to base_dir or absolute)
            use_jsonl: Whether to use JSON Lines format

        Returns:
            List of Query objects
        """
        path = self._resolve_path(path)

        if not path.exists():
            logger.warning(f"File not found: {path}")
            return []

        if use_jsonl:
            data = load_jsonl(path)
        else:
            data = load_json(path)

        queries = []
        failed_count = 0
        for i, item in enumerate(data):
            try:
                queries.append(Query(**item))
            except Exception as e:
                failed_count += 1
                if failed_count <= 3:  # Only log first 3 errors
                    logger.warning(f"Failed to parse query at index {i}: {e}")
                    logger.debug(f"Problematic item: {item}")

        if failed_count > 0:
            logger.warning(f"Failed to parse {failed_count}/{len(data)} queries")

        logger.info(f"Loaded {len(queries)} queries from {path}")

        return queries

    def save_responses(
        self,
        responses: dict[str, list[SearchResponse]],
        path: Path | str,
    ) -> None:
        """Save search responses to file.

        Args:
            responses: Dictionary mapping query_id to list of responses
            path: File path
        """
        path = self._resolve_path(path)

        data = {
            query_id: [r.model_dump() for r in resp_list]
            for query_id, resp_list in responses.items()
        }

        save_json(data, path)
        logger.info(f"Saved responses for {len(responses)} queries to {path}")

    def load_responses(self, path: Path | str) -> dict[str, list[SearchResponse]]:
        """Load search responses from file.

        Args:
            path: File path

        Returns:
            Dictionary mapping query_id to list of responses
        """
        path = self._resolve_path(path)

        if not path.exists():
            logger.warning(f"File not found: {path}")
            return {}

        data = load_json(path)

        responses = {
            query_id: [SearchResponse(**r) for r in resp_list]
            for query_id, resp_list in data.items()
        }

        logger.info(f"Loaded responses for {len(responses)} queries from {path}")
        return responses

    def save_evaluations(
        self,
        results: list[EvaluationResult],
        path: Path | str,
        use_jsonl: bool = True,
    ) -> None:
        """Save evaluation results to file.

        Args:
            results: List of EvaluationResult objects
            path: File path
            use_jsonl: Whether to use JSON Lines format
        """
        path = self._resolve_path(path)

        data = [r.model_dump() for r in results]

        if use_jsonl:
            save_jsonl(data, path)
        else:
            save_json(data, path)

        logger.info(f"Saved {len(results)} evaluation results to {path}")

    def load_evaluations(
        self,
        path: Path | str,
        use_jsonl: bool = True,
    ) -> list[EvaluationResult]:
        """Load evaluation results from file.

        Args:
            path: File path
            use_jsonl: Whether to use JSON Lines format

        Returns:
            List of EvaluationResult objects
        """
        path = self._resolve_path(path)

        if not path.exists():
            logger.warning(f"File not found: {path}")
            return []

        if use_jsonl:
            data = load_jsonl(path)
        else:
            data = load_json(path)

        results = [EvaluationResult(**item) for item in data]
        logger.info(f"Loaded {len(results)} evaluation results from {path}")

        return results

    def append_to_file(self, item: dict, path: Path | str) -> None:
        """Append item to JSON Lines file.

        Args:
            item: Dictionary to append
            path: File path
        """
        path = self._resolve_path(path)
        append_jsonl(item, path)

    def _resolve_path(self, path: Path | str) -> Path:
        """Resolve path relative to base_dir or as absolute.

        Args:
            path: Path to resolve

        Returns:
            Resolved Path object
        """
        path = Path(path)
        # If path is absolute or starts with data/, use as-is
        # This prevents double prefixing (data/data/...)
        if path.is_absolute() or str(path).startswith("data/"):
            return path
        # Otherwise, resolve relative to base_dir
        return self.base_dir / path
