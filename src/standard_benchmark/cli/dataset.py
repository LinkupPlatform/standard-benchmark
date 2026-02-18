"""Dataset loading commands."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from standard_benchmark.storage.json_store import JSONStore
from standard_benchmark.utils.logging_config import setup_logging

app = typer.Typer()
console = Console()


@app.command("load")
def load_dataset(
    source: str = typer.Option(
        ...,
        "--source",
        "-s",
        help="Data source: csv or jsonl",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="File path (for csv/jsonl sources)",
    ),
    output: Path = typer.Option(
        Path("data/queries.jsonl"),
        "--output",
        "-o",
        help="Output file path (JSONL)",
    ),
    text_column: str = typer.Option(
        "text",
        "--text-column",
        help="Column name containing query text (for csv)",
    ),
    id_column: Optional[str] = typer.Option(
        None,
        "--id-column",
        help="Column name containing query IDs (for csv)",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum number of queries to load",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level",
    ),
):
    """Load a dataset from various sources into JSONL format."""
    setup_logging(level=log_level)

    console.print(
        "\n[bold cyan]Standard Benchmark - Dataset Loading[/bold cyan]\n"
    )
    console.print(f"Source: {source}")

    if source == "csv":
        if not path:
            console.print("[red]Error: --path is required for CSV source[/red]")
            raise typer.Exit(1)

        from standard_benchmark.data.loaders.base import BaseLoader
        from standard_benchmark.data.loaders.csv_loader import CSVLoader

        loader: BaseLoader = CSVLoader(
            path=path,
            text_column=text_column,
            id_column=id_column,
        )
        console.print(f"File: {path}")

    elif source == "jsonl":
        if not path:
            console.print("[red]Error: --path is required for JSONL source[/red]")
            raise typer.Exit(1)

        from standard_benchmark.data.loaders.jsonl_loader import JSONLLoader

        loader = JSONLLoader(path=path)
        console.print(f"File: {path}")

    else:
        console.print(
            f"[red]Error: Unknown source '{source}'."
            " Use: csv, jsonl[/red]"
        )
        raise typer.Exit(1)

    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        _task = progress.add_task("Loading dataset...", total=None)
        queries = loader.load()

    if limit:
        queries = queries[:limit]

    # Save results
    store = JSONStore()
    output.parent.mkdir(parents=True, exist_ok=True)
    store.save_queries(queries, output, use_jsonl=True)

    console.print(f"\n✅ [green]Loaded {len(queries)} queries[/green]")
    console.print(f"✅ [green]Saved to {output}[/green]\n")


@app.command("sample")
def load_sample(
    output: Path = typer.Option(
        Path("data/queries.jsonl"),
        "--output",
        "-o",
        help="Output file path (JSONL)",
    ),
):
    """Load built-in sample dataset (8 queries) for smoke testing."""
    console.print("\n[bold cyan]Standard Benchmark - Sample Dataset[/bold cyan]\n")

    from standard_benchmark.data.loaders.sample_loader import SampleLoader

    loader = SampleLoader()
    queries = loader.load()

    store = JSONStore()
    output.parent.mkdir(parents=True, exist_ok=True)
    store.save_queries(queries, output, use_jsonl=True)

    table = Table(title="Sample Queries")
    table.add_column("ID", style="cyan")
    table.add_column("Query", style="white")

    for q in queries:
        table.add_row(q.id, q.text[:70] + "..." if len(q.text) > 70 else q.text)

    console.print(table)
    console.print(
        f"\n✅ [green]Saved {len(queries)} sample queries to {output}[/green]\n"
    )


if __name__ == "__main__":
    app()
