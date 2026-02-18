"""Automated metrics computation."""

import logging
from typing import Optional

import numpy as np
from bert_score import score as bertscore_score
from rouge_score import rouge_scorer

from standard_benchmark.data.models import SearchResponse, Source

logger = logging.getLogger(__name__)


class AutomatedMetrics:
    """Compute automated evaluation metrics."""

    def __init__(self, skip_bertscore: bool = False):
        """Initialize automated metrics calculator.

        Args:
            skip_bertscore: If True, skip BERTScore
                computation (useful if model loading
                fails)
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.skip_bertscore = skip_bertscore

    def compute_all(
        self, query: str, response: SearchResponse, reference: Optional[str] = None
    ) -> dict[str, float]:
        """Compute all automated metrics.

        Args:
            query: Original query
            response: Search response
            reference: Optional reference answer (sources) for faithfulness BERTScore

        Returns:
            Dictionary of metric scores (includes
            descriptive metrics not used in scoring)
        """
        metrics: dict[str, float] = {}

        # Descriptive metrics (computed but NOT included in score calculation)
        metrics["response_length"] = float(len(response.response_text))
        metrics["response_word_count"] = float(len(response.response_text.split()))
        metrics["source_count"] = float(len(response.sources))
        metrics["unique_domains"] = float(
            self.unique_domains(response.sources)
        )  # Descriptive only
        metrics["entity_coverage"] = self.entity_coverage(
            query, response.response_text
        )  # Descriptive only
        metrics["lexical_overlap"] = self.lexical_overlap(
            query, response.response_text
        )

        # BERTScore for relevance (query vs response) - USED IN SCORE
        if not self.skip_bertscore:
            metrics["bertscore_relevance"] = self.bertscore(
                response.response_text, query
            )
        else:
            metrics["bertscore_relevance"] = 0.0  # Default when skipped

        # Metrics that need reference sources for faithfulness - USED IN SCORE
        if reference:
            if not self.skip_bertscore:
                metrics["bertscore_faithfulness"] = self.bertscore(
                    response.response_text, reference
                )
            else:
                metrics["bertscore_faithfulness"] = 0.0  # Default when skipped

            rouge_scores = self.rouge_scores(response.response_text, reference)
            metrics.update({f"rouge_{k}": v for k, v in rouge_scores.items()})

        # Source quality indicators
        if response.sources:
            metrics["avg_snippet_length"] = (
                float(np.mean([
                    len(s.snippet)
                    for s in response.sources
                    if s.snippet
                ]))
                if any(s.snippet for s in response.sources)
                else 0.0
            )

            metrics["sources_with_titles"] = float(
                sum(1 for s in response.sources if s.title)
            ) / len(response.sources)

        return metrics

    def bertscore(self, candidate: str, reference: str, lang: str = "en") -> float:
        """Compute BERTScore F1.

        Args:
            candidate: Generated text
            reference: Reference text
            lang: Language code

        Returns:
            BERTScore F1 score
        """
        try:
            import torch

            # Explicitly set device to avoid meta tensor issues
            device = "cuda" if torch.cuda.is_available() else "cpu"

            P, R, F1 = bertscore_score(
                [candidate],
                [reference],
                lang=lang,
                verbose=False,
                device=device,
                rescale_with_baseline=False,
            )
            return float(F1.mean())
        except RuntimeError as e:
            if "meta tensor" in str(e):
                try:
                    P, R, F1 = bertscore_score(
                        [candidate], [reference], lang=lang, verbose=False
                    )
                    return float(F1.mean())
                except Exception:
                    logger.warning(
                        f"BERTScore computation failed (meta tensor issue): {e}"
                    )
                    return 0.0
            else:
                logger.warning(f"BERTScore computation failed: {e}")
                return 0.0
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            return 0.0

    def rouge_scores(self, candidate: str, reference: str) -> dict[str, float]:
        """Compute ROUGE scores.

        Args:
            candidate: Generated text
            reference: Reference text

        Returns:
            Dictionary with rouge1, rouge2, rougeL F1 scores
        """
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        except Exception as e:
            logger.warning(f"ROUGE computation failed: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def entity_coverage(self, query: str, response: str) -> float:
        """Compute what fraction of query entities appear in response.

        Args:
            query: Query string
            response: Response text

        Returns:
            Coverage score (0-1)
        """
        try:
            import spacy

            # Load spacy model (included as dependency)
            nlp = spacy.load("en_core_web_sm")

            # Extract entities from query
            query_doc = nlp(query)
            query_entities = set(ent.text.lower() for ent in query_doc.ents)

            if not query_entities:
                return 1.0  # No entities to cover

            # Check coverage in response
            response_lower = response.lower()
            covered = sum(1 for ent in query_entities if ent in response_lower)

            return covered / len(query_entities)

        except Exception as e:
            logger.warning(f"Entity coverage computation failed: {e}")
            return 0.0

    def lexical_overlap(self, query: str, response: str) -> float:
        """Compute lexical overlap between query and response.

        Args:
            query: Query string
            response: Response text

        Returns:
            Overlap score (0-1)
        """
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        if not query_words:
            return 0.0

        overlap = query_words.intersection(response_words)
        return len(overlap) / len(query_words)

    def unique_domains(self, sources: list[Source]) -> int:
        """Count unique domains in sources.

        Args:
            sources: List of sources

        Returns:
            Number of unique domains
        """
        domains = set()
        for source in sources:
            if source.domain:
                domains.add(source.domain)
            elif source.url:
                # Extract domain from URL if not provided
                from urllib.parse import urlparse

                try:
                    domain = urlparse(source.url).netloc
                    if domain:
                        domains.add(domain)
                except Exception:
                    pass

        return len(domains)

    def response_length_score(
        self, response: str, ideal_min: int = 100, ideal_max: int = 500
    ) -> float:
        """Score response length (1-10 scale).

        Args:
            response: Response text
            ideal_min: Minimum ideal length
            ideal_max: Maximum ideal length

        Returns:
            Length score (1-10)
        """
        length = len(response)

        if length < ideal_min:
            # Penalize too short
            return max(1, 10 * (length / ideal_min))
        elif length > ideal_max:
            # Penalize too long (but less severely)
            excess = length - ideal_max
            penalty = min(5, excess / 100)
            return max(5, 10 - penalty)
        else:
            # Ideal range
            return 10.0
