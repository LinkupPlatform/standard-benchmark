"""HuggingFace judges library integration for LLM-as-a-Judge evaluation."""

import asyncio
import logging
import os
from typing import Optional

from standard_benchmark.config.settings import settings
from standard_benchmark.data.models import SearchResponse

logger = logging.getLogger(__name__)


class HuggingFaceJudge:
    """HuggingFace judges-based response evaluation using three specialized judges.

    This class integrates three research-backed evaluator prompts:
    1. PollMultihopCorrectness - Binary correctness classifier (True/False)
    2. PrometheusAbsoluteCoarseCorrectness - Fine-grained correctness grader (1-5)
    3. MTBenchChatBotResponseQuality - Response quality evaluator (1-10)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        correctness_classifier_weight: float = 0.45,
        correctness_grader_weight: float = 0.45,
        response_quality_weight: float = 0.1,
    ):
        """Initialize HuggingFace judge with three sub-judges.

        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model to use for evaluation
                (format: "openai/model-name",
                default: "openai/gpt-4o-mini")
            correctness_classifier_weight: Weight for
                PollMultihopCorrectness (default: 0.45)
            correctness_grader_weight: Weight for
                PrometheusAbsoluteCoarseCorrectness
                (default: 0.45)
            response_quality_weight: Weight for
                MTBenchChatBotResponseQuality
                (default: 0.1)
        """
        # Validate weights sum to 1.0
        total_weight = (
            correctness_classifier_weight
            + correctness_grader_weight
            + response_quality_weight
        )
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(
                "Judge weights must sum to 1.0, "
                f"got {total_weight:.3f}"
            )

        self.model = model
        self.correctness_classifier_weight = correctness_classifier_weight
        self.correctness_grader_weight = correctness_grader_weight
        self.response_quality_weight = response_quality_weight

        # Get API key from settings or use passed key
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. "
                "Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Set OPENAI_API_KEY env var for LiteLLM
        # (used by judges library).
        # The judges library uses LiteLLM internally
        # which reads from env vars.
        os.environ["OPENAI_API_KEY"] = self.api_key

        # Initialize judges lazily to avoid import errors if package not installed
        self._judges_initialized = False
        self._correctness_classifier = None
        self._correctness_grader = None
        self._response_quality_evaluator = None

    def _initialize_judges(self):
        """Lazy initialization of judges to handle import errors gracefully."""
        if self._judges_initialized:
            return

        try:
            from judges.classifiers.correctness import (
                PollMultihopCorrectness,
            )
            from judges.graders.correctness import (
                PrometheusAbsoluteCoarseCorrectness,
            )
            from judges.graders.response_quality import (
                MTBenchChatBotResponseQuality,
            )

            # Initialize the three judges
            self._correctness_classifier = (
                PollMultihopCorrectness(model=self.model)
            )
            self._correctness_grader = PrometheusAbsoluteCoarseCorrectness(
                model=self.model
            )
            self._response_quality_evaluator = MTBenchChatBotResponseQuality(
                model=self.model
            )

            self._judges_initialized = True

        except ImportError as e:
            logger.error(
                "Failed to import judges library. "
                "Please install with: "
                "pip install judges[litellm]"
            )
            raise ImportError(
                "judges library not found. "
                "Install with: pip install judges[litellm]"
            ) from e

    async def _run_judge_async(
        self,
        judge,
        input_text: str,
        output: str,
        expected: Optional[str] = None,
    ):
        """Run a judge asynchronously in a thread pool.

        Args:
            judge: The judge instance to run
            input_text: The query/input text
            output: The response/output text
            expected: Optional expected/ground truth answer

        Returns:
            Judgment result with score and reasoning
        """
        loop = asyncio.get_event_loop()

        # Run the synchronous judge.judge() call in thread pool
        if expected:
            result = await loop.run_in_executor(
                None,
                lambda: judge.judge(
                    input=input_text,
                    output=output,
                    expected=expected,
                ),
            )
        else:
            result = await loop.run_in_executor(
                None,
                lambda: judge.judge(
                    input=input_text, output=output
                ),
            )

        return result

    async def evaluate_response_async(
        self,
        query: str,
        response: SearchResponse,
        expected: Optional[str] = None,
    ) -> dict[str, float]:
        """Evaluate a single response asynchronously using all three judges.

        Args:
            query: Original query
            response: Search response to evaluate
            expected: Optional ground truth answer for correctness evaluation

        Returns:
            Dictionary of metric scores (all converted to 1-10 scale)
        """
        self._initialize_judges()

        try:
            # Prepare context from sources (for expected answer if not provided)
            retrieved_contexts = []
            for s in response.sources:
                if s.snippet:
                    retrieved_contexts.append(f"{s.title or 'Untitled'}: {s.snippet}")
                else:
                    url = getattr(s, 'url', 'unknown')
                    retrieved_contexts.append(f"{s.title or 'Untitled'}: No source snippet available for {url}")

            if not retrieved_contexts:
                retrieved_contexts = ["No sources were provided for this response."]

            # Use top 3 sources as reference context if no expected answer
            if not expected:
                expected = " ".join(retrieved_contexts[:3])

            # Run all three judges concurrently
            logger.info(
                "Running 3 HuggingFace judges "
                f"for query: {query[:50]}..."
            )

            # Create tasks for concurrent execution
            tasks = [
                self._run_judge_async(
                    self._correctness_classifier,
                    query,
                    response.response_text,
                    expected,
                ),
                self._run_judge_async(
                    self._correctness_grader, query, response.response_text, expected
                ),
                self._run_judge_async(
                    self._response_quality_evaluator,
                    query,
                    response.response_text,
                    None,  # Quality evaluator doesn't need expected answer
                ),
            ]

            # Wait for all judges to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            (
                classifier_result_raw,
                grader_result_raw,
                quality_result_raw,
            ) = results

            # Handle any exceptions - create fallback objects with score/reasoning
            class _FallbackResult:
                def __init__(
                    self, score: object, reasoning: str
                ) -> None:
                    self.score = score
                    self.reasoning = reasoning

            if isinstance(classifier_result_raw, BaseException):
                logger.error(
                    "Correctness classifier failed: "
                    f"{classifier_result_raw}"
                )
                classifier_result = _FallbackResult(
                    score=False, reasoning="Error"
                )
            else:
                classifier_result = classifier_result_raw
            if isinstance(grader_result_raw, BaseException):
                logger.error(
                    "Correctness grader failed: "
                    f"{grader_result_raw}"
                )
                grader_result = _FallbackResult(
                    score=3, reasoning="Error"
                )
            else:
                grader_result = grader_result_raw
            if isinstance(quality_result_raw, BaseException):
                logger.error(
                    "Quality evaluator failed: "
                    f"{quality_result_raw}"
                )
                quality_result = _FallbackResult(
                    score=5, reasoning="Error"
                )
            else:
                quality_result = quality_result_raw

            # Extract scores
            # 1. Correctness Classifier
            #    (Binary: True/False -> 0 or 10)
            # True = 10 (correct), False = 0 (incorrect)
            classifier_score_10 = (
                10.0 if classifier_result.score else 0.0
            )

            # 2. Correctness Grader
            #    (1-5 Likert scale from Prometheus paper)
            # PrometheusAbsoluteCoarseCorrectness uses
            # a 5-point Likert scale:
            # 1=Completely incorrect,
            # 2=Mostly incorrect,
            # 3=Partially correct,
            # 4=Mostly correct,
            # 5=Completely correct
            grader_score_raw = float(
                grader_result.score  # type: ignore[arg-type]
            )
            grader_score_10 = self._convert_rubric_to_10_scale(
                grader_score_raw, max_score=5
            )

            # 3. Response Quality
            #    (1-10 scale - already in correct scale)
            quality_score_raw = float(
                quality_result.score  # type: ignore[arg-type]
            )
            quality_score_10 = max(
                1.0, min(10.0, quality_score_raw)
            )

            # Calculate weighted overall score
            overall_score = (
                classifier_score_10 * self.correctness_classifier_weight
                + grader_score_10 * self.correctness_grader_weight
                + quality_score_10 * self.response_quality_weight
            )

            scores = {
                "hf_correctness_classifier": classifier_score_10,
                "hf_correctness_grader": grader_score_10,
                "hf_response_quality": quality_score_10,
                "hf_overall_quality": overall_score,
            }

            logger.info(
                "HuggingFace scores: "
                f"classifier={classifier_score_10:.2f}"
                f", grader={grader_score_10:.2f}"
                f", quality={quality_score_10:.2f}"
                f", overall={overall_score:.2f}"
            )

            # Log reasoning (optional, for debugging)
            logger.debug(
                "Classifier reasoning: "
                f"{classifier_result.reasoning}"
            )
            logger.debug(
                "Grader reasoning: "
                f"{grader_result.reasoning}"
            )
            logger.debug(
                "Quality reasoning: "
                f"{quality_result.reasoning}"
            )

            return scores

        except Exception as e:
            logger.error(f"HuggingFace evaluation failed: {e}")
            logger.exception("Full traceback:")
            raise

    def evaluate_response(
        self,
        query: str,
        response: SearchResponse,
        expected: Optional[str] = None,
    ) -> dict[str, float]:
        """Synchronous wrapper for evaluate_response_async.

        Args:
            query: Original query
            response: Search response to evaluate
            expected: Optional ground truth answer

        Returns:
            Dictionary of metric scores
        """
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(
                self.evaluate_response_async(
                    query, response, expected
                )
            )
        except Exception as e:
            logger.error(
                "Synchronous HuggingFace "
                f"evaluation failed: {e}"
            )
            raise

    def _convert_to_10_scale(self, score: float) -> float:
        """Convert 0-1 scale to 0-10 scale (for non-binary metrics).

        Args:
            score: Score in 0-1 range

        Returns:
            Score in 0-10 range
        """
        # Map 0-1 to 0-10: score * 10
        return min(10.0, max(0.0, score * 10.0))

    def _convert_rubric_to_10_scale(
        self, score: float, max_score: int = 5
    ) -> float:
        """Convert rubric scale (e.g., 1-5) to 1-10 scale.

        Args:
            score: Score in 1-max_score range
            max_score: Maximum score value (default: 5)

        Returns:
            Score in 1-10 range
        """
        # Map 1-max_score to 1-10:
        #   (score - 1) * (9 / (max_score - 1)) + 1
        if max_score == 1:
            return 5.5  # Middle value
        scaled = (score - 1) * (9.0 / (max_score - 1)) + 1.0
        return min(10.0, max(1.0, scaled))

