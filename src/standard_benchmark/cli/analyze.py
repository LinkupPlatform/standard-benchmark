"""Analysis and dashboard commands."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table


app = typer.Typer()
console = Console()


@app.command("dashboard")
def launch_dashboard(
    port: int = typer.Option(
        8501,
        "--port",
        "-p",
        help="Port to run dashboard on",
    ),
):
    """Launch Streamlit dashboard."""
    import subprocess

    dashboard_path = Path(__file__).parent.parent / "visualization" / "dashboard.py"

    console.print(f"\n[cyan]Launching dashboard on port {port}...[/cyan]")
    console.print("[dim]Dashboard will open in your browser[/dim]\n")

    try:
        subprocess.run(
            ["streamlit", "run", str(dashboard_path), "--server.port", str(port)]
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")


@app.command("stats")
def show_stats(
    evaluations: Path = typer.Option(
        ...,
        "--evaluations",
        "-e",
        help="Input evaluations file (JSON)",
    ),
):
    """Show evaluation statistics."""
    console.print("\n[bold cyan]Evaluation Statistics[/bold cyan]\n")

    with open(evaluations, "r") as f:
        eval_data = json.load(f)

    # Collect scores
    system_scores: dict[str, list[float]] = {}
    for query_id, systems in eval_data.items():
        for system, eval_dict in systems.items():
            if system not in system_scores:
                system_scores[system] = []
            score = eval_dict.get("scores", {}).get("final_score", 0)
            system_scores[system].append(score)

    # Display table
    stats_table = Table(title="System Statistics")
    stats_table.add_column("System", style="cyan")
    stats_table.add_column("Avg Score", style="green", justify="right")
    stats_table.add_column("Min", style="yellow", justify="right")
    stats_table.add_column("Max", style="yellow", justify="right")
    stats_table.add_column("Std Dev", style="blue", justify="right")
    stats_table.add_column("Queries", style="dim", justify="right")

    import numpy as np

    for system in sorted(system_scores.keys()):
        scores = system_scores[system]

        stats_table.add_row(
            system,
            f"{np.mean(scores):.2f}",
            f"{min(scores):.2f}",
            f"{max(scores):.2f}",
            f"{np.std(scores):.2f}",
            str(len(scores)),
        )

    console.print(stats_table)
    console.print()


if __name__ == "__main__":
    app()
