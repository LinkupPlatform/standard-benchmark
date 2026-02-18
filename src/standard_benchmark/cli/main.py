"""Main Typer application with subcommands."""

import typer
from rich.console import Console

from standard_benchmark import __version__
from standard_benchmark.cli import dataset, search, evaluate, analyze

app = typer.Typer(
    name="standard-benchmark",
    help="Standard Benchmark — Open-Source Search API Evaluation Framework",
    add_completion=False,
)

console = Console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"standard-benchmark version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
):
    """Standard Benchmark — Open-Source Search API Evaluation Framework."""
    pass


# Register subcommands
app.add_typer(dataset.app, name="dataset", help="Dataset loading commands")
app.add_typer(search.app, name="search", help="Search execution commands")
app.add_typer(evaluate.app, name="evaluate", help="Evaluation pipeline commands")
app.add_typer(analyze.app, name="analyze", help="Analysis and dashboard commands")


if __name__ == "__main__":
    app()
