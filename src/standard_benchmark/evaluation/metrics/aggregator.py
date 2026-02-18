"""Compute three independent evaluation scores."""

import logging
from typing import Optional


logger = logging.getLogger(__name__)


class ScoreAggregator:
    """Compute three independent evaluation scores.

    Ragas, HuggingFace, and Automated.
    """

    def __init__(
        self,
        ragas_weights: Optional[dict[str, float]] = None,
        huggingface_weights: Optional[dict[str, float]] = None,
        automated_weights: Optional[dict[str, float]] = None,
    ):
        """Initialize score aggregator.

        Args:
            ragas_weights: Weights for Ragas metrics
                (faithfulness, completeness, source_quality)
            huggingface_weights: Weights for HuggingFace metrics
            automated_weights: Weights for automated metrics
                (bertscore_relevance, bertscore_faithfulness)
        """
        # Default weights if not provided
        self.ragas_weights = ragas_weights or {
            "faithfulness": 0.45,
            "completeness": 0.45,
            "source_quality": 0.1,
        }
        self.huggingface_weights = huggingface_weights or {
            "correctness_classifier": 0.45,
            "correctness_grader": 0.45,
            "response_quality": 0.1,
        }
        self.automated_weights = automated_weights or {
            "bertscore_relevance": 0.55,
            "bertscore_faithfulness": 0.45,
        }

        # Normalize weights
        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize all weight dictionaries to sum to 1.0."""
        for weights_dict in [
            self.ragas_weights,
            self.huggingface_weights,
            self.automated_weights,
        ]:
            total = sum(weights_dict.values())
            if total > 0:
                for key in weights_dict:
                    weights_dict[key] /= total

    def compute_all_scores(
        self,
        ragas_scores: Optional[dict[str, float]] = None,
        huggingface_scores: Optional[dict[str, float]] = None,
        automated_scores: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """Compute all three independent scores.

        Args:
            ragas_scores: Ragas evaluation scores dict
            huggingface_scores: HuggingFace evaluation scores dict
            automated_scores: Automated metrics dict

        Returns:
            Dictionary with three scores:
                ragas_score, huggingface_score,
                automated_score (each 1-10)
        """
        scores = {}

        # Only compute scores for systems that were actually run
        if ragas_scores:
            scores["ragas_score"] = self._compute_ragas_score(ragas_scores)

        if huggingface_scores:
            scores["huggingface_score"] = self._compute_huggingface_score(
                huggingface_scores
            )

        if automated_scores:
            scores["automated_score"] = (
                self._compute_automated_score(automated_scores)
            )

        return scores

    def _compute_ragas_score(self, scores: dict[str, float]) -> float:
        """Compute weighted Ragas score.

        Args:
            scores: Dictionary of Ragas metric scores

        Returns:
            Weighted score (1-10)
        """
        score = 0.0
        for metric, weight in self.ragas_weights.items():
            if metric in scores:
                score += scores[metric] * weight
            else:
                score += 5.0 * weight  # Default to middle score if missing

        return score

    def _compute_huggingface_score(self, scores: dict[str, float]) -> float:
        """Compute weighted HuggingFace score.

        Args:
            scores: Dictionary of HuggingFace metric scores

        Returns:
            Weighted score (1-10)
        """
        score = 0.0
        hf_metrics = {
            "correctness_classifier": "hf_correctness_classifier",
            "correctness_grader": "hf_correctness_grader",
            "response_quality": "hf_response_quality",
        }

        for weight_key, score_key in hf_metrics.items():
            weight = self.huggingface_weights.get(weight_key, 0.0)
            if score_key in scores:
                score += scores[score_key] * weight
            else:
                score += 5.0 * weight  # Default to middle score if missing

        return score

    def _compute_automated_score(self, scores: dict[str, float]) -> float:
        """Compute weighted automated metrics score.

        Args:
            scores: Dictionary of automated metric scores

        Returns:
            Weighted score (1-10)
        """
        score = 0.0

        for metric, weight in self.automated_weights.items():
            metric_key = f"automated_{metric}"
            if metric_key in scores:
                value = scores[metric_key]
                # BERTScore is 0-1, convert to 1-10
                if metric.startswith("bertscore_"):
                    value = value * 9 + 1  # Scale 0-1 to 1-10
                score += value * weight
            else:
                score += 5.0 * weight  # Default to middle score if missing

        return score

    def compute_ranking_score(self, rank: int, total_systems: int) -> float:
        """Convert ranking to score (0-10 scale).

        Args:
            rank: Rank position (1 = best)
            total_systems: Total number of systems

        Returns:
            Score (0-10)
        """
        if total_systems <= 1:
            return 10.0

        # Linear scale: 1st place = 10, last place = 4
        # Formula: 10 - (rank - 1) * (6 / (total_systems - 1))
        score = 10 - (rank - 1) * (6 / (total_systems - 1))

        return max(4.0, min(10.0, score))
