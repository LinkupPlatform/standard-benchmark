"""Search execution commands."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.table import Table

from standard_benchmark.search.executor import SearchExecutor
from standard_benchmark.storage.json_store import JSONStore
from standard_benchmark.utils.logging_config import setup_logging

app = typer.Typer()
console = Console()


@app.command("run")
def run_search(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input queries file (JSONL)",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output responses file (JSON)",
    ),
    systems: str = typer.Option(
        "linkup,exa,tavily,perplexity",
        "--systems",
        "-s",
        help="Comma-separated list of systems to use",
    ),
    batch_size: int = typer.Option(
        10,
        "--batch-size",
        "-b",
        help="Number of queries to process in each batch",
    ),
    use_cache: bool = typer.Option(
        True,
        "--cache/--no-cache",
        help="Use response caching",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level",
    ),
):
    """Execute searches across all systems for queries in input file."""
    setup_logging(level=log_level)

    console.print("\n[bold cyan]Standard Benchmark - Search Execution[/bold cyan]\n")

    # Parse systems
    system_list = [s.strip() for s in systems.split(",")]

    # Load queries
    store = JSONStore()
    queries = store.load_queries(input, use_jsonl=True)

    if not queries:
        console.print("[red]No queries found in input file[/red]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(queries)} queries from {input}")
    console.print(f"Systems: {', '.join(system_list)}")
    console.print(f"Batch size: {batch_size}")
    console.print(f"Cache: {'enabled' if use_cache else 'disabled'}\n")

    # Initialize executor
    try:
        executor = SearchExecutor(use_cache=use_cache)
        available_systems = executor.get_available_systems()

        # Filter to available systems
        system_list = [s for s in system_list if s in available_systems]

        if not system_list:
            console.print(
                "[red]No valid search systems available. Check API keys in .env[/red]"
            )
            raise typer.Exit(1)

        console.print(f"Available systems: {', '.join(available_systems)}")
        console.print(f"Using systems: {', '.join(system_list)}\n")

    except Exception as e:
        console.print(f"[red]Failed to initialize executor: {e}[/red]")
        raise typer.Exit(1)

    # Execute searches asynchronously
    async def run_searches():
        all_responses = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Executing searches...", total=len(queries))

            for i, query in enumerate(queries):
                progress.update(
                    task,
                    description=f"Query {i + 1}/{len(queries)}: {query.text[:40]}...",
                )

                responses = await executor.execute_single_query(
                    query, systems=system_list
                )
                all_responses[query.id] = responses

                progress.advance(task)

        return all_responses

    # Run the async function
    all_responses = asyncio.run(run_searches())

    # Convert to serializable format
    output_data = {}
    for query_id, responses in all_responses.items():
        output_data[query_id] = {
            system: response.model_dump() for system, response in responses.items()
        }

    # Save results
    output.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # Display summary
    success_counts = {system: 0 for system in system_list}
    error_counts = {system: 0 for system in system_list}

    for responses in all_responses.values():
        for system, response in responses.items():
            if response.metadata.get("status") == "failed":
                error_counts[system] += 1
            else:
                success_counts[system] += 1

    summary_table = Table(title="Execution Summary")
    summary_table.add_column("System", style="cyan")
    summary_table.add_column("Success", style="green", justify="right")
    summary_table.add_column("Errors", style="red", justify="right")
    summary_table.add_column("Success Rate", style="yellow", justify="right")

    for system in system_list:
        success = success_counts[system]
        errors = error_counts[system]
        total = success + errors
        rate = (success / total * 100) if total > 0 else 0

        summary_table.add_row(system, str(success), str(errors), f"{rate:.1f}%")

    console.print()
    console.print(summary_table)
    console.print(f"\n✅ [green]Saved responses to {output}[/green]\n")


@app.command("single")
def search_single(
    query: str = typer.Option(
        ...,
        "--query",
        "-q",
        help="Query to search",
    ),
    systems: str = typer.Option(
        "linkup,exa,tavily,perplexity",
        "--systems",
        "-s",
        help="Comma-separated list of systems to use",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output file (JSON)",
    ),
):
    """Execute a single query across specified systems."""
    console.print("\n[bold cyan]Single Query Search[/bold cyan]\n")
    console.print(f"Query: [yellow]{query}[/yellow]\n")

    # Parse systems
    system_list = [s.strip() for s in systems.split(",")]

    # Initialize executor
    try:
        executor = SearchExecutor()
        available_systems = executor.get_available_systems()
        system_list = [s for s in system_list if s in available_systems]

        if not system_list:
            console.print("[red]No valid search systems available[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Failed to initialize executor: {e}[/red]")
        raise typer.Exit(1)

    # Create temporary query object
    from standard_benchmark.data.models import Query

    query_obj = Query(
        id="temp",
        text=query,
    )

    # Execute searches
    console.print("Executing searches...\n")
    responses = asyncio.run(
        executor.execute_single_query(query_obj, systems=system_list)
    )

    # Display results
    for system, response in responses.items():
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"[bold]{system.upper()}[/bold]")
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

        if response.metadata.get("status") == "failed":
            console.print(f"[red]Error: {response.metadata.get('error')}[/red]")
        else:
            console.print("\n[green]Answer:[/green]")
            console.print(
                response.response_text[:500] + "..."
                if len(response.response_text) > 500
                else response.response_text
            )
            console.print(f"\n[yellow]Sources ({len(response.sources)}):[/yellow]")
            for i, source in enumerate(response.sources[:5], 1):
                console.print(f"{i}. {source.url}")
            console.print(f"\n[dim]Latency: {response.latency_ms:.0f}ms[/dim]")

        console.print()

    # Save if output specified
    if output:
        output_data = {
            system: response.model_dump() for system, response in responses.items()
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        console.print(f"✅ [green]Saved to {output}[/green]\n")


@app.command("clear-cache")
def clear_cache(
    expired_only: bool = typer.Option(
        False,
        "--expired-only",
        help="Only clear expired entries",
    ),
):
    """Clear response cache."""
    from standard_benchmark.storage.cache import ResponseCache

    cache = ResponseCache()

    if expired_only:
        cleared = cache.clear_expired()
        console.print(f"✅ Cleared {cleared} expired cache entries")
    else:
        cleared = cache.clear_all()
        console.print(f"✅ Cleared {cleared} cache entries")


if __name__ == "__main__":
    app()
