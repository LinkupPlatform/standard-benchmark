"""Custom evaluation metrics with OpenAI structured outputs."""

import asyncio
import logging
from typing import Optional, List

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from standard_benchmark.config.settings import settings
from standard_benchmark.data.models import SearchResponse

logger = logging.getLogger(__name__)


# Structured output schema - verdicts, completeness, and source quality
class CombinedEvaluationOutput(BaseModel):
    """Combined output for faithfulness verdicts, completeness,
    and source quality scores."""

    faithfulness_verdicts: List[int] = Field(
        ...,
        description=(
            "List of faithfulness verdicts (0 or 1) for each"
            " statement you extracted. 1 means the statement"
            " can be directly inferred from the context,"
            " 0 means it cannot."
        ),
    )
    completeness_score: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Completeness score from 1-5."
            " 1=severely incomplete,"
            " 2=addresses some aspects,"
            " 3=covers main points but lacks depth,"
            " 4=mostly complete,"
            " 5=thoroughly complete"
        ),
    )
    source_quality_score: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Source quality score from 1-5. How relevant/useful"
            " are the provided sources for answering the"
            " query? 1=completely irrelevant sources,"
            " 2=tangentially related, 3=somewhat relevant,"
            " 4=mostly relevant, 5=highly relevant sources"
            " that enable comprehensive answer"
        ),
    )


class RagasJudge:
    """Ragas-based response evaluation using standardized metrics."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        """Initialize Ragas judge.

        Args:
            api_key: OpenAI API key (defaults to settings)
            model: OpenAI model to use
        """
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required for Ragas judging"
            )

        # Use AsyncOpenAI client for all metrics
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model

    async def _evaluate_all_in_one(
        self, query: str, response_text: str, context: str
    ) -> CombinedEvaluationOutput:
        """Extract statements internally, evaluate faithfulness,
        completeness, and relevancy in ONE API call.

        The model extracts statements internally but only
        returns verdicts and scores to minimize output
        tokens.

        Args:
            query: The original query
            response_text: The response to evaluate
            context: The context (source material) to evaluate against

        Returns:
            Combined output with faithfulness verdicts,
            completeness score, and relevancy score
        """
        prompt = (
            "You are evaluating a response to a query"
            " based on retrieved context/sources.\n\n"
            f"**Query:** {query}\n\n"
            f"**Response:** {response_text}\n\n"
            f"**Context (Retrieved Sources):** {context}"
            "\n\n---\n\n"
            "Please perform the following tasks:\n\n"
            "1. **Extract Statements** (internally, don't"
            " return them): Break down the response into"
            " simple, atomic factual statements. Each"
            " statement should be:\n"
            "   - A single factual claim\n"
            "   - Self-contained and independent\n"
            "   - Verifiable against source material\n\n"
            "2. **Faithfulness Verdicts** (return this):"
            " For EACH extracted statement, determine if"
            " it can be DIRECTLY inferred from the"
            " context/sources. Return 1 if the statement"
            " is supported by the sources, 0 if it is"
            " not or cannot be verified.\n"
            "   - Give 0 if the statement contradicts the"
            " sources or adds information not present in"
            " sources\n"
            "   - Give 0 if the statement is not mentioned"
            " in the sources at all (hallucination)\n"
            "   - Give 1 only if the statement is clearly"
            " supported by the provided sources\n"
            "   - Give 1 if the response explicitly says"
            ' "I don\'t know" or "cannot be answered'
            ' based on sources"\n\n'
            "3. **Completeness Score** (return this):"
            " Evaluate how completely the response"
            " addresses the specific query asked"
            " (1-5):\n"
            "   - Consider: Can the query be fully"
            " answered given the sources provided?\n"
            "   - 1 = Severely incomplete, missing most"
            " key information needed to answer the"
            " query\n"
            "   - 2 = Addresses only some aspects, with"
            " significant information gaps\n"
            "   - 3 = Covers main points but lacks depth"
            " or misses some relevant details\n"
            "   - 4 = Mostly complete, only minor details"
            " missing\n"
            "   - 5 = Thoroughly complete, comprehensively"
            " addresses all aspects of the query\n"
            "   - NOTE: If sources are poor/irrelevant but"
            " response does its best, score based on"
            " what's answerable\n\n"
            "4. **Source Quality Score** (return this):"
            " Evaluate how relevant and useful the"
            " provided sources are for answering the"
            " query (1-5):\n"
            "   - Judge ONLY the sources themselves, not"
            " how well the response uses them\n"
            "   - This measures retrieval quality - do the"
            " sources contain the right information?\n"
            "   - 1 = Completely irrelevant sources,"
            " contain no information related to the"
            " query\n"
            "   - 2 = Tangentially related sources, some"
            " connection but missing key information\n"
            "   - 3 = Somewhat relevant sources, contain"
            " partial information to answer the query\n"
            "   - 4 = Mostly relevant sources, contain"
            " most information needed with minor gaps\n"
            "   - 5 = Highly relevant sources, contain"
            " comprehensive information to fully answer"
            " the query\n\n"
            "ALIGNMENT CHECK - Use all scores together"
            " to diagnose:\n"
            "- Good Retrieval + Good Generation: High"
            " source_quality (4-5) + High faithfulness"
            " (mostly 1s) + High completeness (4-5)\n"
            "- Bad Retrieval + Hallucination: Low"
            " source_quality (1-2) + Low faithfulness"
            " (mostly 0s)\n"
            "- Good Retrieval + Poor Generation: High"
            " source_quality (4-5) + Low faithfulness"
            " (mostly 0s)\n"
            "- Bad Retrieval + Honest Response: Low"
            " source_quality (1-2) + High faithfulness"
            ' (1 if says "can\'t answer") + Low'
            " completeness (1-2)\n\n"
            "Return ONLY the faithfulness verdicts array,"
            " completeness score, and source quality"
            " score. Do NOT return the statements"
            " themselves."
        )

        try:
            completion = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert evaluator"
                            " of response quality."
                            " Extract statements"
                            " internally, provide"
                            " accurate faithfulness"
                            " verdicts, completeness"
                            " score, and source quality"
                            " score. Return only verdicts"
                            " and scores."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=CombinedEvaluationOutput,
            )

            result = completion.choices[0].message.parsed
            if not result:
                raise ValueError("No parsed result from OpenAI")

            return result

        except Exception as e:
            logger.error(f"Failed to evaluate all-in-one: {e}")
            raise

    async def evaluate_response_async(
        self, query: str, response: SearchResponse
    ) -> dict[str, float]:
        """Evaluate a single response asynchronously using combined metrics.

        Args:
            query: Original query
            response: Search response to evaluate

        Returns:
            Dictionary of metric scores (converted to 1-10 scale)
        """
        try:
            # Prepare contexts from sources
            retrieved_contexts = []
            for s in response.sources:
                if s.snippet:
                    retrieved_contexts.append(f"{s.title or 'Untitled'}: {s.snippet}")
                else:
                    url = getattr(s, 'url', 'unknown')
                    retrieved_contexts.append(f"{s.title or 'Untitled'}: No source snippet available for {url}")

            if not retrieved_contexts:
                retrieved_contexts = ["No sources were provided for this response."]

            # Prepare reference context (concatenate all snippets)
            context = " ".join(retrieved_contexts[:3])  # Use top 3 sources

            # Step 1: Evaluate faithfulness, completeness,
            # relevancy AND source quality in ONE API call
            logger.info("Evaluating all metrics (all-in-one)...")
            result = await self._evaluate_all_in_one(
                query=query, response_text=response.response_text, context=context
            )
            verdicts = result.faithfulness_verdicts
            supported = sum(verdicts)
            total = len(verdicts)
            logger.info(
                f"All-in-one evaluation complete: "
                f"{supported}/{total} statements"
                f" supported, "
                f"completeness="
                f"{result.completeness_score}/5, "
                f"source_quality="
                f"{result.source_quality_score}/5"
            )

            # Step 2: Calculate faithfulness score manually from verdicts
            if result.faithfulness_verdicts:
                faithfulness_raw = sum(result.faithfulness_verdicts) / len(
                    result.faithfulness_verdicts
                )
            else:
                faithfulness_raw = 0.0

            # Extract scores and convert to 1-10 scale
            scores = {}

            # 1. Faithfulness (0-1 scale) - from verdicts
            scores["faithfulness"] = (
                self._convert_to_10_scale(faithfulness_raw)
            )
            faith_score = scores["faithfulness"]
            logger.debug(
                f"Faithfulness: {faithfulness_raw:.3f}"
                f" -> {faith_score:.2f}/10"
                f" ({supported}/{total}"
                f" statements supported)"
            )

            # 2. Completeness (1-5 scale) - from OpenAI
            scores["completeness"] = (
                self._convert_rubric_to_10_scale(
                    float(result.completeness_score)
                )
            )
            comp_score = scores["completeness"]
            logger.debug(
                f"Completeness:"
                f" {result.completeness_score:.1f}/5"
                f" -> {comp_score:.2f}/10"
            )

            # 4. Source Quality (1-5 scale) - from OpenAI
            scores["source_quality"] = (
                self._convert_rubric_to_10_scale(
                    float(result.source_quality_score)
                )
            )
            sq_score = scores["source_quality"]
            logger.debug(
                f"Source Quality:"
                f" {result.source_quality_score:.1f}/5"
                f" -> {sq_score:.2f}/10"
            )

            # Overall quality (weighted average:
            # faith=0.45, comp=0.45, source_qual=0.1)
            scores["overall_quality"] = (
                scores["faithfulness"] * 0.45
                + scores["completeness"] * 0.45
                + scores["source_quality"] * 0.1
            )

            logger.info(
                f"Evaluation scores:"
                f" Faithfulness={faith_score:.1f}"
                f" (45%),"
                f" Completeness={comp_score:.1f}"
                f" (45%),"
                f" Source Quality={sq_score:.1f}"
                f" (10%),"
                f" Overall="
                f"{scores['overall_quality']:.1f}"
            )

            return scores

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Re-raise so the pipeline can properly
            # count it as a failure
            raise

    def evaluate_response(
        self, query: str, response: SearchResponse
    ) -> dict[str, float]:
        """Synchronous wrapper for evaluate_response_async.

        Args:
            query: Original query
            response: Search response to evaluate

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
                self.evaluate_response_async(query, response)
            )
        except Exception as e:
            logger.error(f"Synchronous evaluation failed: {e}")
            raise

    def _convert_to_10_scale(self, score: float) -> float:
        """Convert Ragas 0-1 scale to 1-10 scale.

        Args:
            score: Score in 0-1 range

        Returns:
            Score in 1-10 range
        """
        # Map 0-1 to 1-10: score * 9 + 1
        return min(10.0, max(1.0, score * 9.0 + 1.0))

    def _convert_rubric_to_10_scale(self, score: float) -> float:
        """Convert Ragas RubricsScore 1-5 scale to 1-10 scale.

        Args:
            score: Score in 1-5 range

        Returns:
            Score in 1-10 range
        """
        # Map 1-5 to 1-10: (score - 1) * (9/4) + 1
        # 1 -> 1, 3 -> 5.5, 5 -> 10
        return min(10.0, max(1.0, (score - 1) * 2.25 + 1.0))

