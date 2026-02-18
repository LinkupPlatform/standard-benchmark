"""End-to-end evaluation orchestrator."""

import asyncio
import logging
from typing import Optional

from standard_benchmark.data.models import (
    Query,
    SearchResponse,
    EvaluationResult,
)
from standard_benchmark.evaluation.metrics.automated import (
    AutomatedMetrics,
)
from standard_benchmark.evaluation.metrics.ragas_judge import (
    RagasJudge,
)
from standard_benchmark.evaluation.metrics.huggingface_judge import (
    HuggingFaceJudge,
)
from standard_benchmark.evaluation.metrics.aggregator import (
    ScoreAggregator,
)
from standard_benchmark.storage.evaluation_cache import EvaluationCache
from standard_benchmark.evaluation.progress import EvaluationProgress

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Orchestrate end-to-end evaluation process."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        hf_model: str = "openai/gpt-4o-mini",
        use_automated: bool = True,
        # Enable BERTScore by default
        skip_bertscore: bool = False,
        # Use Ragas metrics (default: True)
        use_ragas: bool = True,
        # Use HuggingFace judges (default: False)
        use_huggingface: bool = False,
        hf_correctness_classifier_weight: float = 0.45,
        hf_correctness_grader_weight: float = 0.45,
        hf_response_quality_weight: float = 0.1,
        max_concurrent_llm: int = 5,
        max_concurrent_queries: int = 2,
        use_cache: bool = True,
    ):
        """Initialize evaluation pipeline.

        Args:
            openai_api_key: OpenAI API key for both
                Ragas and HuggingFace judges
            llm_model: Model to use for Ragas evaluation
                (default: gpt-4o-mini)
            hf_model: Model for HuggingFace judges
                (format: "openai/model-name",
                default: "openai/gpt-4o-mini")
            use_automated: Whether to compute automated
                metrics
            skip_bertscore: Skip BERTScore computation
                (default: False)
            use_ragas: Whether to use Ragas metrics
                (default: True)
            use_huggingface: Whether to use HuggingFace
                judges (default: False)
            hf_correctness_classifier_weight: Weight for
                HF correctness classifier (default: 0.45)
            hf_correctness_grader_weight: Weight for HF
                correctness grader (default: 0.45)
            hf_response_quality_weight: Weight for HF
                response quality (default: 0.1)
            max_concurrent_llm: Max concurrent LLM API
                calls (default: 5)
            max_concurrent_queries: Max concurrent queries
                to evaluate (default: 2)
            use_cache: Whether to cache evaluation
                results (default: True)
        """
        self.use_automated = use_automated
        self.use_ragas = use_ragas
        self.use_huggingface = use_huggingface
        self.max_concurrent_llm = max_concurrent_llm
        self.max_concurrent_queries = max_concurrent_queries
        self.llm_model = llm_model
        self.hf_model = hf_model

        # Initialize cache
        self.cache = EvaluationCache() if use_cache else None

        # Initialize components
        if use_automated:
            self.automated_metrics = AutomatedMetrics(
                skip_bertscore=skip_bertscore
            )

        if use_ragas:
            self.ragas_judge = RagasJudge(
                api_key=openai_api_key, model=llm_model
            )

        if use_huggingface:
            self.huggingface_judge = HuggingFaceJudge(
                api_key=openai_api_key,
                model=hf_model,
                correctness_classifier_weight=hf_correctness_classifier_weight,
                correctness_grader_weight=hf_correctness_grader_weight,
                response_quality_weight=hf_response_quality_weight,
            )

        self.aggregator = ScoreAggregator()

    async def evaluate_query(
        self,
        query: Query,
        responses: dict[str, SearchResponse],
        llm_semaphore: asyncio.Semaphore,
        progress: Optional[EvaluationProgress] = None,
    ) -> dict[str, EvaluationResult]:
        """Evaluate all responses for a single query asynchronously.

        Args:
            query: Query object
            responses: Dictionary mapping system name to SearchResponse
            llm_semaphore: Semaphore to control concurrent LLM calls
            progress: Optional progress tracker

        Returns:
            Dictionary mapping system name to EvaluationResult
        """
        results = {}

        # Cache configuration
        cache_config = {
            "model": self.llm_model,
            "hf_model": self.hf_model,
            "use_automated": self.use_automated,
            "use_ragas": self.use_ragas,
            "use_huggingface": self.use_huggingface,
        }

        # HuggingFace Judges Evaluation:
        # Evaluate each response independently (CONCURRENTLY)
        huggingface_scores = {}
        huggingface_cache_status = {}  # Track which metrics were cached per system
        if self.use_huggingface:

            async def evaluate_huggingface(system: str, response: SearchResponse):
                logger.debug(
                    f"[{system}] Starting HuggingFace "
                    f"judges evaluation for query: "
                    f"{query.text[:50]}..."
                )

                # HuggingFace metrics to evaluate
                hf_metrics = [
                    "hf_correctness_classifier",
                    "hf_correctness_grader",
                    "hf_response_quality",
                    "hf_overall_quality",
                ]
                scores = {}
                cached_metrics = []

                # Check cache for all metrics at once
                if self.cache:
                    logger.debug(
                        f"[{system}] Checking cache for HuggingFace metrics..."
                    )
                    cached_scores = self.cache.get_metrics(
                        query.text,
                        response.response_text,
                        system,
                        cache_config,
                        hf_metrics,
                    )
                    for metric, cached_score in cached_scores.items():
                        if cached_score is not None:
                            scores[metric] = cached_score
                            cached_metrics.append(metric)
                            logger.debug(
                                f"[{system}] Found "
                                f"cached {metric}: "
                                f"{cached_score:.2f}"
                            )

                # Check if all metrics are cached
                if len(cached_metrics) == len(hf_metrics):
                    logger.debug(
                        f"[{system}] All HF "
                        f"metrics cached!"
                    )
                    return (
                        system,
                        scores,
                        cached_metrics,
                        True,
                    )  # Return cached=True flag

                # Some or all metrics need evaluation
                cached_ct = len(cached_metrics)
                total_ct = len(hf_metrics)
                logger.debug(
                    f"[{system}] Cached: "
                    f"{cached_ct}/{total_ct} metrics"
                )
                logger.debug(
                    f"[{system}] Waiting for "
                    f"LLM semaphore..."
                )

                async with llm_semaphore:
                    logger.info(
                        f"[{system}] Acquired "
                        f"semaphore, calling "
                        f"HuggingFace judges..."
                    )
                    try:
                        # Evaluate all metrics
                        new_scores = (
                            await self.huggingface_judge.evaluate_response_async(
                                query.text, response
                            )
                        )
                        logger.info(
                            f"[{system}] HuggingFace "
                            f"evaluation complete! "
                            f"Scores: {new_scores}"
                        )

                        # Cache all newly computed metrics (batch operation)
                        if self.cache:
                            metrics_to_cache = {
                                metric: score
                                for metric, score in new_scores.items()
                                if metric not in cached_metrics
                            }
                            if metrics_to_cache:
                                self.cache.set_metrics(
                                    query.text,
                                    response.response_text,
                                    system,
                                    cache_config,
                                    metrics_to_cache,
                                )
                                n = len(metrics_to_cache)
                                logger.debug(
                                    f"[{system}] Cached "
                                    f"{n} new HF metrics"
                                )

                        # Use newly computed scores
                        scores = new_scores

                        return (
                            system,
                            scores,
                            cached_metrics,
                            False,
                        )  # Return cached=False flag

                    except Exception as e:
                        logger.error(
                            f"[{system}] HuggingFace "
                            f"evaluation failed: {e}"
                        )
                        logger.exception(
                            f"[{system}] Full traceback:"
                        )
                        raise

            tasks = [
                evaluate_huggingface(sys, resp)
                for sys, resp in responses.items()
            ]
            results_list = await asyncio.gather(*tasks)

            # Unpack results: scores and cache status (don't update progress yet)
            for system, scores, cached_metrics, was_cached in results_list:
                huggingface_scores[system] = scores
                huggingface_cache_status[system] = {
                    "cached_metrics": cached_metrics,
                    "total_metrics": len(scores),
                    "was_cached": was_cached,
                }

        # Ragas Evaluation: Evaluate each response independently (CONCURRENTLY)
        # Now with granular per-metric caching
        ragas_scores = {}
        ragas_cache_status = {}  # Track which metrics were cached per system
        if self.use_ragas:

            async def evaluate_ragas(system: str, response: SearchResponse):
                logger.debug(
                    f"[{system}] Starting Ragas "
                    f"evaluation for query: "
                    f"{query.text[:50]}..."
                )

                # RAGAS metrics to evaluate
                ragas_metrics = [
                    "faithfulness",
                    "completeness",
                    "source_quality",
                    "overall_quality",
                ]
                scores = {}
                cached_metrics = []

                # Check cache for all metrics at once
                if self.cache:
                    logger.debug(
                        f"[{system}] Checking cache "
                        f"for RAGAS metrics..."
                    )
                    cached_scores = self.cache.get_metrics(
                        query.text,
                        response.response_text,
                        system,
                        cache_config,
                        ragas_metrics,
                    )
                    for metric, cached_score in cached_scores.items():
                        if cached_score is not None:
                            scores[metric] = cached_score
                            cached_metrics.append(metric)
                            logger.debug(
                                f"[{system}] Found "
                                f"cached {metric}: "
                                f"{cached_score:.2f}"
                            )

                # Check if all metrics are cached
                if len(cached_metrics) == len(ragas_metrics):
                    logger.debug(
                        f"[{system}] All Ragas "
                        f"metrics cached!"
                    )
                    return (
                        system,
                        scores,
                        cached_metrics,
                        True,
                    )  # Return cached=True flag

                # Some or all metrics need evaluation
                cached_ct = len(cached_metrics)
                total_ct = len(ragas_metrics)
                logger.debug(
                    f"[{system}] Cached: "
                    f"{cached_ct}/{total_ct} metrics"
                )
                logger.debug(
                    f"[{system}] Waiting for "
                    f"LLM semaphore..."
                )

                async with llm_semaphore:
                    logger.info(
                        f"[{system}] Acquired "
                        f"semaphore, calling "
                        f"Ragas judge..."
                    )
                    try:
                        # Evaluate all metrics
                        # (Ragas doesn't support partial eval)
                        new_scores = await self.ragas_judge.evaluate_response_async(
                            query.text, response
                        )
                        logger.info(
                            f"[{system}] Ragas "
                            f"evaluation complete! "
                            f"Scores: {new_scores}"
                        )

                        # Cache all newly computed metrics (batch operation)
                        if self.cache:
                            metrics_to_cache = {
                                metric: score
                                for metric, score in new_scores.items()
                                if metric not in cached_metrics
                            }
                            if metrics_to_cache:
                                self.cache.set_metrics(
                                    query.text,
                                    response.response_text,
                                    system,
                                    cache_config,
                                    metrics_to_cache,
                                )
                                n = len(metrics_to_cache)
                                logger.debug(
                                    f"[{system}] Cached "
                                    f"{n} new metrics"
                                )

                        # Use newly computed scores
                        # (overwrite cached for consistency)
                        scores = new_scores

                        return (
                            system,
                            scores,
                            cached_metrics,
                            False,
                        )  # Return cached=False flag

                    except Exception as e:
                        logger.error(
                            f"[{system}] Ragas "
                            f"evaluation failed: {e}"
                        )
                        logger.exception(
                            f"[{system}] Full traceback:"
                        )
                        raise

            tasks = [
                evaluate_ragas(sys, resp)
                for sys, resp in responses.items()
            ]
            results_list = await asyncio.gather(*tasks)

            # Unpack results: scores and cache status (don't update progress yet)
            for system, scores, cached_metrics, was_cached in results_list:
                ragas_scores[system] = scores
                ragas_cache_status[system] = {
                    "cached_metrics": cached_metrics,
                    "total_metrics": len(scores),
                    "was_cached": was_cached,
                }

        # Automated metrics: Compute for each response (CONCURRENTLY)
        automated_scores = {}
        if self.use_automated:

            async def compute_automated(system: str, response: SearchResponse):
                # Check cache first
                if self.cache:
                    cached = self.cache.get(
                        query.text, response.response_text, system, cache_config
                    )
                    if cached:
                        # Extract automated scores from cached result
                        cached_auto = {
                            k.replace("automated_", ""): v
                            for k, v in cached.scores.items()
                            if k.startswith("automated_")
                        }
                        return system, cached_auto

                # Run sync function in thread pool (faster than sequential)
                loop = asyncio.get_event_loop()
                scores = await loop.run_in_executor(
                    None, self.automated_metrics.compute_all, query.text, response
                )
                return system, scores

            tasks = [compute_automated(sys, resp) for sys, resp in responses.items()]
            results_list = await asyncio.gather(*tasks)
            automated_scores = dict(results_list)

        # Update progress once per system after all judges complete
        if progress:
            for system in responses.keys():
                # Determine overall cache status for this system
                hf_cached = (
                    huggingface_cache_status.get(system, {}).get("was_cached", False)
                    if self.use_huggingface
                    else True
                )
                ragas_cached = (
                    ragas_cache_status.get(system, {}).get("was_cached", False)
                    if self.use_ragas
                    else True
                )

                # Only mark as cached if ALL enabled judges were cached
                all_cached = hf_cached and ragas_cached

                # Check for partial caching
                hf_partial = (
                    len(
                        huggingface_cache_status.get(system, {}).get(
                            "cached_metrics", []
                        )
                    )
                    > 0
                    if self.use_huggingface
                    else False
                )
                ragas_partial = (
                    len(ragas_cache_status.get(system, {}).get("cached_metrics", []))
                    > 0
                    if self.use_ragas
                    else False
                )
                has_partial = (hf_partial or ragas_partial) and not all_cached

                progress.update_system(
                    system,
                    cached=all_cached,
                    partial_cached=has_partial,
                    failed=False,
                )

        # Aggregate scores for each system
        for system, response in responses.items():
            # Get Ragas scores
            ragas = ragas_scores.get(system, {}) if self.use_ragas else {}

            # Get HuggingFace scores
            huggingface = (
                huggingface_scores.get(system, {}) if self.use_huggingface else {}
            )

            # Get automated scores
            automated = automated_scores.get(system, {}) if self.use_automated else {}

            # Compute final score with priority:
            # Compute three independent scores using aggregator
            independent_scores = self.aggregator.compute_all_scores(
                ragas_scores=ragas,
                huggingface_scores=huggingface,
                automated_scores={f"automated_{k}": v for k, v in automated.items()},
            )

            # Compute final_score as average of scores that were actually computed
            enabled_scores = [
                v for k, v in independent_scores.items()
                if k.endswith("_score")
            ]
            final_score = (
                sum(enabled_scores) / len(enabled_scores) if enabled_scores else 0.0
            )

            # Create evaluation result with all scores
            all_scores = {
                "final_score": final_score,
                **independent_scores,  # ragas_score, huggingface_score, automated_score
                **ragas,  # faithfulness, completeness, source_quality, overall_quality
                # hf_correctness_classifier, hf_correctness_grader,
                # hf_response_quality, hf_overall_quality
                **huggingface,
                **{f"automated_{k}": v for k, v in automated.items()},
            }

            result = EvaluationResult(
                query_id=query.id or "",
                system=system,
                scores=all_scores,
                labels=[],
                metadata={
                    "query_text": query.text,
                    "response_length": len(response.response_text),
                    "source_count": len(response.sources),
                    "latency_ms": response.latency_ms,
                },
            )

            results[system] = result

            # Log cache status for this system
            if self.use_ragas and system in ragas_cache_status:
                cache_info = ragas_cache_status[system]
                n_cached = len(cache_info['cached_metrics'])
                n_total = cache_info['total_metrics']
                logger.debug(
                    f"[{system}] Ragas metrics: "
                    f"{n_cached}/{n_total} cached"
                )

            if self.use_huggingface and system in huggingface_cache_status:
                cache_info = huggingface_cache_status[system]
                n_cached = len(cache_info['cached_metrics'])
                n_total = cache_info['total_metrics']
                logger.debug(
                    f"[{system}] HuggingFace "
                    f"metrics: "
                    f"{n_cached}/{n_total} cached"
                )

        return results

    async def evaluate_batch(
        self,
        queries: list[Query],
        all_responses: dict[str, dict[str, SearchResponse]],
        show_progress: bool = True,
    ) -> list[dict[str, EvaluationResult]]:
        """Evaluate multiple queries concurrently.

        Args:
            queries: List of Query objects
            all_responses: Dictionary mapping query_id to dict of system responses
            show_progress: Whether to show progress table (default: True)

        Returns:
            List of evaluation results for each query
        """
        # Get list of systems from first query's responses
        first_query_id = queries[0].id if queries else None
        if first_query_id:
            systems = list(
                all_responses.get(first_query_id, {}).keys()
            )
        else:
            systems = []

        # Create progress tracker
        progress = None
        if show_progress and systems:
            progress = EvaluationProgress(total_queries=len(queries), systems=systems)
            progress.start()

        # Create semaphores for rate limiting
        llm_semaphore = asyncio.Semaphore(self.max_concurrent_llm)
        query_semaphore = asyncio.Semaphore(self.max_concurrent_queries)

        async def evaluate_with_limit(query: Query):
            async with query_semaphore:
                if query.id not in all_responses:
                    logger.warning(f"No responses found for query {query.id}")
                    return None
                return await self.evaluate_query(
                    query, all_responses[query.id], llm_semaphore, progress
                )

        try:
            # Execute all queries concurrently with rate limiting
            tasks = [evaluate_with_limit(q) for q in queries]
            results = await asyncio.gather(*tasks)
        finally:
            if progress:
                progress.stop()
                progress.print_summary()

        # Filter out None results
        return [r for r in results if r is not None]
