"""Chart generation using matplotlib and seaborn."""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def plot_score_distribution(
    evaluations: dict[str, dict[str, dict]],
    systems: list[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot score distribution across systems.

    Args:
        evaluations: Dict of query_id -> system -> evaluation dict
        systems: List of system names
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    # Collect scores
    data = []
    for query_id, sys_evals in evaluations.items():
        for system in systems:
            if system in sys_evals:
                score = sys_evals[system].get("scores", {}).get("final_score", 0)
                data.append({"System": system, "Score": score})

    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="System", y="Score", ax=ax)
    ax.set_title("Score Distribution by System", fontsize=16, fontweight="bold")
    ax.set_ylabel("Final Score", fontsize=12)
    ax.set_xlabel("System", fontsize=12)
    ax.set_ylim(0, 10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_metric_heatmap(
    evaluations: dict[str, dict[str, dict]],
    systems: list[str],
    metrics: list[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot heatmap of metrics across systems.

    Args:
        evaluations: Dict of query_id -> system -> evaluation dict
        systems: List of system names
        metrics: List of metric names to plot
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    # Collect metric averages
    data = []
    for system in systems:
        system_metrics = []
        for metric in metrics:
            scores = []
            for query_id, sys_evals in evaluations.items():
                if system in sys_evals:
                    score = sys_evals[system].get("scores", {}).get(metric, 0)
                    scores.append(score)
            avg_score = np.mean(scores) if scores else 0
            system_metrics.append(avg_score)
        data.append(system_metrics)

    df = pd.DataFrame(data, index=systems, columns=metrics)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=10,
        ax=ax,
        cbar_kws={"label": "Score"},
    )
    ax.set_title("Metric Heatmap by System", fontsize=16, fontweight="bold")
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Systems", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_score_comparison(
    evaluations: dict[str, dict[str, dict]],
    systems: list[str],
    score_key: str = "final_score",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot score comparison across all systems.

    Args:
        evaluations: Dict of query_id -> system -> evaluation dict
        systems: List of system names
        score_key: Which score to compare
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    # Collect scores per system
    data = []
    for query_id, sys_evals in evaluations.items():
        for system in systems:
            if system in sys_evals:
                score = sys_evals[system].get("scores", {}).get(score_key, 0)
                data.append({"System": system, "Score": score, "Query": query_id})

    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="System", y="Score", ax=ax)
    ax.set_title(
        f"Score Comparison ({score_key})", fontsize=16, fontweight="bold"
    )
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("System", fontsize=12)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
