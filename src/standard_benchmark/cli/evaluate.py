"""Evaluation commands."""

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from standard_benchmark.evaluation.pipeline import EvaluationPipeline
from standard_benchmark.storage.json_store import JSONStore
from standard_benchmark.utils.logging_config import setup_logging
from standard_benchmark.data.models import SearchResponse

app = typer.Typer()
console = Console()


@app.command("run")
def run_evaluation(
    queries: Path = typer.Option(
        ...,
        "--queries",
        "-q",
        help="Input queries file (JSONL)",
    ),
    responses: Path = typer.Option(
        ...,
        "--responses",
        "-r",
        help="Input responses file (JSON from search run)",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output evaluations file (JSON)",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="OpenAI model for Ragas LLM judging",
    ),
    hf_model: str = typer.Option(
        "openai/gpt-4o-mini",
        "--hf-model",
        help=(
            "HuggingFace judges model"
            " (format: provider/model-name,"
            " e.g. openai/gpt-4o-mini)"
        ),
    ),
    skip_automated: bool = typer.Option(
        False,
        "--skip-automated",
        help="Skip automated metrics",
    ),
    use_ragas: bool = typer.Option(
        True,
        "--use-ragas/--no-ragas",
        help="Use Ragas metrics (faithfulness, completeness, source_quality)",
    ),
    use_huggingface: bool = typer.Option(
        False,
        "--use-huggingface/--no-huggingface",
        help=(
            "Use HuggingFace judges"
            " (correctness classifier,"
            " grader, response quality)"
        ),
    ),
    hf_classifier_weight: float = typer.Option(
        0.45,
        "--hf-classifier-weight",
        help="Weight for HF correctness classifier within HuggingFace judges",
    ),
    hf_grader_weight: float = typer.Option(
        0.45,
        "--hf-grader-weight",
        help="Weight for HF correctness grader within HuggingFace judges",
    ),
    hf_quality_weight: float = typer.Option(
        0.1,
        "--hf-quality-weight",
        help="Weight for HF response quality within HuggingFace judges",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level",
    ),
    log_file: Path = typer.Option(
        None,
        "--log-file",
        help="Path to log file (useful for debugging while progress table displays)",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="Disable progress display (for debugging)",
    ),
):
    """Run complete evaluation pipeline on search responses."""
    setup_logging(level=log_level, log_file=log_file)

    console.print("\n[bold cyan]Standard Benchmark - Evaluation Pipeline[/bold cyan]\n")

    # Load queries
    store = JSONStore()
    query_list = store.load_queries(queries, use_jsonl=True)

    if not query_list:
        console.print("[red]No queries found[/red]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(query_list)} queries")

    # Load responses
    with open(responses, "r") as f:
        responses_data = json.load(f)

    console.print(f"Loaded responses for {len(responses_data)} queries")

    # Convert to SearchResponse objects
    all_responses = {}
    for query_id, systems in responses_data.items():
        all_responses[query_id] = {
            system: SearchResponse(**resp_data) for system, resp_data in systems.items()
        }

    # Initialize pipeline
    console.print("Initializing evaluation pipeline...")
    console.print(f"  LLM Model: {model} (OpenAI)")
    if use_ragas and use_huggingface:
        console.print(f"  - Ragas judge: {model}")
        console.print(f"  - HuggingFace judges: {hf_model}")
    console.print(f"  Automated metrics: {'disabled' if skip_automated else 'enabled'}")
    console.print(f"  Ragas evaluation: {'enabled' if use_ragas else 'disabled'}")
    console.print(
        f"  HuggingFace evaluation: {'enabled' if use_huggingface else 'disabled'}"
    )
    if use_huggingface:
        console.print(f"    - Classifier weight: {hf_classifier_weight}")
        console.print(f"    - Grader weight: {hf_grader_weight}")
        console.print(f"    - Quality weight: {hf_quality_weight}")
    console.print()

    try:
        pipeline = EvaluationPipeline(
            llm_model=model,
            hf_model=hf_model,
            use_automated=not skip_automated,
            use_ragas=use_ragas,
            use_huggingface=use_huggingface,
            hf_correctness_classifier_weight=hf_classifier_weight,
            hf_correctness_grader_weight=hf_grader_weight,
            hf_response_quality_weight=hf_quality_weight,
            max_concurrent_llm=5,  # 5 concurrent LLM API calls
            max_concurrent_queries=2,  # 2 queries evaluated at once
        )
    except Exception as e:
        console.print(f"[red]Failed to initialize pipeline: {e}[/red]")
        raise typer.Exit(1)

    # Run evaluation asynchronously
    console.print(
        "\n[yellow]Running async evaluation"
        " (queries and systems in parallel)"
        "...[/yellow]\n"
    )

    if log_file:
        console.print(f"[dim]Logging to: {log_file}[/dim]\n")

    # Run async evaluation (progress display handled
    # by EvaluationProgress inside pipeline)
    results_list = asyncio.run(
        pipeline.evaluate_batch(
            query_list, all_responses, show_progress=not no_progress
        )
    )

    # Convert to dict format
    all_evaluations = {}
    for query, results in zip(query_list, results_list):
        if results:
            all_evaluations[query.id] = results

    # Convert to serializable format
    output_data = {}
    for query_id, results in all_evaluations.items():
        output_data[query_id] = {
            system: result.model_dump() for system, result in results.items()
        }

    # Save results
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # Display summary
    system_scores: dict[str, list[float]] = {}
    for results in all_evaluations.values():
        for system, result in results.items():
            if system not in system_scores:
                system_scores[system] = []
            system_scores[system].append(float(result.scores.get("final_score", 0)))

    summary_table = Table(title="Evaluation Summary")
    summary_table.add_column("System", style="cyan")
    summary_table.add_column("Avg Score", style="green", justify="right")
    summary_table.add_column("Min", style="yellow", justify="right")
    summary_table.add_column("Max", style="yellow", justify="right")
    summary_table.add_column("Queries", style="dim", justify="right")

    for system in sorted(system_scores.keys()):
        scores = system_scores[system]
        summary_table.add_row(
            system,
            f"{sum(scores) / len(scores):.2f}",
            f"{min(scores):.2f}",
            f"{max(scores):.2f}",
            str(len(scores)),
        )

    console.print()
    console.print(summary_table)
    console.print(f"\n✅ [green]Saved evaluations to {output}[/green]\n")


@app.command("single")
def evaluate_single(
    query: str = typer.Option(
        ...,
        "--query",
        "-q",
        help="Query text",
    ),
    responses_file: Path = typer.Option(
        ...,
        "--responses",
        "-r",
        help="Responses JSON file (from search single)",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="OpenAI model for LLM judging",
    ),
):
    """Evaluate responses for a single query."""
    console.print("\n[bold cyan]Single Query Evaluation[/bold cyan]\n")
    console.print(f"Query: [yellow]{query}[/yellow]\n")

    # Load responses
    with open(responses_file, "r") as f:
        responses_data = json.load(f)

    responses = {
        system: SearchResponse(**resp_data)
        for system, resp_data in responses_data.items()
    }

    # Create temporary query object
    from standard_benchmark.data.models import Query

    query_obj = Query(
        id="temp",
        text=query,
    )

    # Initialize pipeline
    console.print("Initializing evaluation pipeline...")
    pipeline = EvaluationPipeline(llm_model=model, max_concurrent_llm=5)

    # Run evaluation asynchronously
    console.print("Evaluating responses (async)...\n")
    results = asyncio.run(
        pipeline.evaluate_query(query_obj, responses, asyncio.Semaphore(10))
    )

    # Display results
    results_table = Table(title="Evaluation Results")
    results_table.add_column("System", style="cyan")
    results_table.add_column("Final Score", style="green", justify="right")
    results_table.add_column("Ragas Avg", style="yellow", justify="right")
    results_table.add_column("Automated Avg", style="yellow", justify="right")

    for system, result in sorted(
        results.items(), key=lambda x: x[1].scores["final_score"], reverse=True
    ):
        # Calculate Ragas average
        ragas_keys = ["faithfulness", "completeness", "source_quality"]
        ragas_scores = [
            float(result.scores.get(k, 0)) for k in ragas_keys if k in result.scores
        ]
        ragas_avg = sum(ragas_scores) / len(ragas_scores) if ragas_scores else 0.0

        # Calculate automated avg
        automated_keys = [k for k in result.scores.keys() if k.startswith("automated_")]
        automated_avg = (
            sum(float(result.scores[k]) for k in automated_keys) / len(automated_keys)
            if automated_keys
            else 0.0
        )

        results_table.add_row(
            system,
            f"{result.scores['final_score']:.2f}",
            f"{ragas_avg:.2f}",
            f"{automated_avg:.2f}",
        )

    console.print(results_table)
    console.print()


@app.command("clear-cache")
def clear_evaluation_cache(
    expired_only: bool = typer.Option(
        False,
        "--expired-only",
        help="Only clear expired cache entries",
    ),
):
    """Clear evaluation cache."""
    from standard_benchmark.storage.evaluation_cache import EvaluationCache

    console.print("\n[bold cyan]Evaluation Cache Management[/bold cyan]\n")

    cache = EvaluationCache()

    # Show stats first
    stats = cache.get_stats()
    console.print(f"Total entries: {stats['total_entries']}")
    console.print(f"Expired entries: {stats['expired_entries']}")
    console.print(f"Cache size: {stats['total_size_mb']:.2f} MB")
    console.print(f"Cache location: {stats['cache_dir']}\n")

    # Clear cache
    if expired_only:
        console.print("[yellow]Clearing expired entries...[/yellow]")
        cache.clear_expired()
    else:
        console.print("[yellow]Clearing all cache entries...[/yellow]")
        cache.clear_all()

    # Show stats after
    stats = cache.get_stats()
    console.print("\n✅ [green]Cache cleared[/green]")
    console.print(f"Remaining entries: {stats['total_entries']}\n")


if __name__ == "__main__":
    app()
