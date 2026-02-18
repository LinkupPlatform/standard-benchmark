"""Progress tracking for evaluation pipeline."""

import logging
import time
from typing import Dict
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.logging import RichHandler


class EvaluationProgress:
    """Track and display evaluation progress."""

    def __init__(self, total_queries: int, systems: list[str]):
        """Initialize progress tracker.

        Args:
            total_queries: Total number of queries to evaluate
            systems: List of system names
        """
        self.total_queries = total_queries
        self.systems = systems
        self.console = Console()

        # Track progress per system
        self.completed: Dict[str, int] = {sys: 0 for sys in systems}
        self.cached: Dict[str, int] = {sys: 0 for sys in systems}
        self.partial_cached: Dict[str, int] = {
            sys: 0 for sys in systems
        }  # New: track partial caching
        self.failed: Dict[str, int] = {sys: 0 for sys in systems}

        # Timing
        self.start_time = time.time()

        # Live display
        self.live = None

        # Store original logging handlers to restore later
        self.original_handlers = None
        self.rich_handler = None

    def start(self):
        """Start the live display and configure logging to work with Rich."""
        # Save original logging configuration
        root_logger = logging.getLogger()
        self.original_handlers = root_logger.handlers.copy()

        # Remove existing handlers temporarily
        for handler in self.original_handlers:
            root_logger.removeHandler(handler)

        # Add Rich handler that works with Live display
        self.rich_handler = RichHandler(
            console=self.console,
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
        # Only show WARNING and above during progress display to reduce clutter
        self.rich_handler.setLevel(logging.WARNING)
        root_logger.addHandler(self.rich_handler)

        # Start live display
        self.live = Live(
            self._generate_table(), console=self.console, refresh_per_second=4
        )
        self.live.start()

    def stop(self):
        """Stop the live display and restore original logging configuration."""
        if self.live:
            self.live.stop()

        # Restore original logging handlers
        if self.original_handlers is not None:
            root_logger = logging.getLogger()

            # Remove Rich handler
            if self.rich_handler:
                root_logger.removeHandler(self.rich_handler)

            # Restore original handlers
            for handler in self.original_handlers:
                root_logger.addHandler(handler)

    def update_system(
        self,
        system: str,
        cached: bool = False,
        partial_cached: bool = False,
        failed: bool = False,
    ):
        """Update progress for a system.

        Args:
            system: System name
            cached: Whether result was fully cached
            partial_cached: Whether result was partially cached (some metrics cached)
            failed: Whether evaluation failed
        """
        if system not in self.completed:
            return

        if cached:
            self.cached[system] += 1
        elif partial_cached:
            self.partial_cached[system] += 1
        elif failed:
            self.failed[system] += 1
        else:
            self.completed[system] += 1

        self._refresh()

    def _refresh(self):
        """Refresh the display."""
        if self.live:
            self.live.update(self._generate_table())

    def _generate_table(self) -> Table:
        """Generate the progress table.

        Returns:
            Rich Table object
        """
        table = Table(
            title="🔍 Evaluation Progress", show_header=True, header_style="bold cyan"
        )

        table.add_column("System", style="bold", width=12)
        table.add_column("Completed", justify="right", width=10)
        table.add_column("Cached", justify="right", style="green", width=10)
        table.add_column("Partial", justify="right", style="yellow", width=10)
        table.add_column("Failed", justify="right", style="red", width=10)
        table.add_column("Progress", width=30)

        for system in self.systems:
            total_done = (
                self.completed[system]
                + self.cached[system]
                + self.partial_cached[system]
                + self.failed[system]
            )
            progress = (
                (total_done / self.total_queries * 100) if self.total_queries > 0 else 0
            )

            # Create progress bar
            bar_width = 20
            filled = int(bar_width * progress / 100)
            bar = "█" * filled + "░" * (bar_width - filled)

            table.add_row(
                system.upper(),
                str(self.completed[system]),
                str(self.cached[system]),
                str(self.partial_cached[system]),
                str(self.failed[system]),
                f"{bar} {progress:.0f}%",
            )

        # Add summary row
        elapsed = time.time() - self.start_time
        total_completed = (
            sum(self.completed.values())
            + sum(self.cached.values())
            + sum(self.partial_cached.values())
        )

        table.add_row("", "", "", "", "", "", style="dim")

        # Estimate time remaining
        if total_completed > 0:
            avg_time_per_eval = elapsed / total_completed
            remaining_evals = len(self.systems) * self.total_queries - total_completed
            eta_seconds = avg_time_per_eval * remaining_evals
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
        else:
            eta_str = "calculating..."

        summary = (
            f"⏱️  Elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s  |  ETA: {eta_str}"
        )
        table.caption = summary

        return table

    def print_summary(self):
        """Print final summary."""
        self.console.print()
        self.console.print("[bold green]✅ Evaluation Complete![/bold green]")
        self.console.print()

        # Create summary table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("System", style="bold")
        table.add_column("Total", justify="right")
        table.add_column("Completed", justify="right", style="green")
        table.add_column("Cached", justify="right", style="yellow")
        table.add_column("Partial", justify="right", style="cyan")
        table.add_column("Failed", justify="right", style="red")

        for system in self.systems:
            total = (
                self.completed[system]
                + self.cached[system]
                + self.partial_cached[system]
                + self.failed[system]
            )
            table.add_row(
                system.upper(),
                str(total),
                str(self.completed[system]),
                str(self.cached[system]),
                str(self.partial_cached[system]),
                str(self.failed[system]),
            )

        self.console.print(table)

        # Total time
        elapsed = time.time() - self.start_time
        self.console.print(
            f"\n⏱️  Total time: {int(elapsed // 60)}m {int(elapsed % 60)}s"
        )
        self.console.print()
