# src/ui/components/progress.py

from rich.progress  import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.live      import Live
from rich.panel     import Panel
from rich.text      import Text
from src.ui.theme   import console, COLORS, BORDER_STYLE, PANEL_PADDING, SYMBOLS


# -------------------------------------------------------------------------
# Progress Bar Column Layout
# -------------------------------------------------------------------------

def _build_progress() -> Progress:
    """
    Construct the Progress instance with the column layout.

    Columns (left → right):
        spinner     — animated activity indicator
        task label  — current step description
        bar         — visual fill bar
        m of n      — e.g. "2/3 batches"
        percentage  — e.g. "66%"
        elapsed     — time since pipeline start
    """
    return Progress(
        SpinnerColumn(
            spinner_name="dots2",
            style=f"bold {COLORS['primary']}",
            finished_text=f"[success]{SYMBOLS['success']}[/]",
        ),
        TextColumn(
            "[step]{task.description}[/]",
            justify="left",
        ),
        BarColumn(
            bar_width=32,
            style=COLORS["muted"],
            complete_style=COLORS["primary"],
            finished_style=COLORS["success"],
            pulse_style=COLORS["secondary"],
        ),
        MofNCompleteColumn(
            separator=" of ",
        ),
        TaskProgressColumn(
            style=f"bold {COLORS['secondary']}",
        ),
        TimeElapsedColumn(),
        console=console,
        transient=False,   # keep progress visible after completion
    )


# -------------------------------------------------------------------------
# Pipeline Progress Manager
# -------------------------------------------------------------------------

class PipelineProgress:
    """
    Manages the Rich progress bar for the batch generation pipeline.

    Wraps Rich's Progress + Live into a clean context manager so
    interface.py never touches Rich internals directly.

    Usage:
        with PipelineProgress(total_batches=3) as progress:
            for batch in batches:
                # do work
                progress.advance(description="Batch 2/3 — requesting 2 variations")

            progress.complete()   # mark done
            # or
            progress.fail()       # mark failed
    """

    def __init__(self, total_batches: int, total_variations: int) -> None:
        """
        Args:
            total_batches    (int): Total number of batches in this pipeline run.
            total_variations (int): Total variations requested (used for panel title).
        """
        self.total_batches     = total_batches
        self.total_variations  = total_variations
        self._progress         = _build_progress()
        self._task_id          = None
        self._live             = None
        self._batches_done     = 0

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> "PipelineProgress":
        self._task_id = self._progress.add_task(
            description = f"{SYMBOLS['batch']}  Initializing pipeline...",
            total       = self.total_batches,
        )

        panel = Panel(
            self._progress,
            title   = f"[primary] {SYMBOLS['batch']} BATCH PIPELINE [/]",
            subtitle= f"[muted] {self.total_variations} variations · {self.total_batches} batches [/]",
            border_style = BORDER_STYLE,
            padding      = PANEL_PADDING,
        )

        self._live = Live(
            panel,
            console     = console,
            refresh_per_second = 12,
            transient   = False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)

    # -------------------------------------------------------------------------
    # Public Controls — called by interface.py during the pipeline loop
    # -------------------------------------------------------------------------

    def advance(self, batch_num: int, batch_size: int, model: str) -> None:
        """
        Advance the progress bar by one batch and update the description.

        Args:
            batch_num  (int): Current batch number (1-based).
            batch_size (int): How many variations this batch is requesting.
            model      (str): Model being used for this batch.
        """
        self._batches_done += 1
        short_model = model.split("/")[-1]    # "nvidia/nemotron-..." → "nemotron-..."

        self._progress.update(
            self._task_id,
            advance     = 1,
            description = (
                f"{SYMBOLS['batch']}  Batch [batch]{batch_num}/{self.total_batches}[/] "
                f"{SYMBOLS['arrow']} [success]{batch_size} variation(s)[/] "
                f"via [model]{short_model}[/]"
            ),
        )

    def set_description(self, description: str) -> None:
        """
        Update the task description without advancing — useful for
        mid-batch status updates like 'retrying...' or 'switching model'.

        Args:
            description (str): New description string (plain text, no markup needed).
        """
        self._progress.update(
            self._task_id,
            description = f"{SYMBOLS['batch']}  {description}",
        )

    def complete(self) -> None:
        """
        Mark the pipeline as fully complete.
        Fills the bar to 100% and updates description to a success state.
        """
        self._progress.update(
            self._task_id,
            completed   = self.total_batches,
            description = (
                f"[success]{SYMBOLS['success']}  Pipeline complete — "
                f"{self.total_variations} variations generated[/]"
            ),
        )

    def fail(self, reason: str = "Pipeline failed") -> None:
        """
        Mark the pipeline as failed.
        Stops the bar and updates description to an error state.

        Args:
            reason (str): Short failure reason shown in the progress bar.
        """
        self._progress.update(
            self._task_id,
            description = (
                f"[error]{SYMBOLS['error']}  {reason}[/]"
            ),
        )

    def partial(self, collected: int) -> None:
        """
        Mark the pipeline as partially complete — some batches succeeded,
        at least one failed.

        Args:
            collected (int): Number of variations successfully collected.
        """
        self._progress.update(
            self._task_id,
            description = (
                f"[warning]{SYMBOLS['warning']}  Partial result — "
                f"{collected}/{self.total_variations} variations collected[/]"
            ),
        )