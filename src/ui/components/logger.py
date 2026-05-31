# src/ui/components/logger.py

from datetime       import datetime
from rich.text      import Text
from rich.rule      import Rule
from rich.panel     import Panel
from src.ui.theme   import console, COLORS, BORDER_STYLE, PANEL_PADDING, SYMBOLS


# -------------------------------------------------------------------------
# Internal Helpers
# -------------------------------------------------------------------------

def _timestamp() -> str:
    """Current time formatted as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


def _format_event(
    symbol:     str,
    style:      str,
    label:      str,
    message:    str,
    detail:     str | None = None,
) -> Text:
    """
    Build a single log line as a Rich Text object.

    Format:
        HH:MM:SS  ✦ LABEL    message  [detail]

    Args:
        symbol  (str): Icon from SYMBOLS dict.
        style   (str): Rich style tag for symbol + label.
        label   (str): Short uppercase category tag e.g. 'BATCH', 'MODEL'.
        message (str): Main log message.
        detail  (str): Optional secondary detail shown dimmed at the end.
    """
    line = Text()
    line.append(f" {_timestamp()} ", style="muted")
    line.append(f" {symbol} ",       style=style)
    line.append(f"{label:<10}",      style=f"bold {COLORS['muted']}")
    line.append(f"{message}",        style="text")

    if detail:
        line.append(f"  {SYMBOLS['arrow']} ", style="muted")
        line.append(f"{detail}",              style="muted")

    return line


# -------------------------------------------------------------------------
# Event Logger
# -------------------------------------------------------------------------

class EventLogger:
    """
    Terminal event logger for the pipeline UI.

    Prints color-coded, timestamped log lines to the shared console.
    Covers every pipeline event: batch start/end, model switching,
    retries, failures, saves, and section dividers.

    All methods are stateless — just print and return.
    No file I/O — that's handled by src/utils/logger.py.

    Usage:
        logger = EventLogger()
        logger.batch_start(batch_num=1, total=3, size=2, start_index=1)
        logger.batch_success(batch_num=1, received=2, model="nemotron...")
        logger.retry(attempt=2, total=3, model="nemotron...", error="timeout")
        logger.model_fallback(primary="nemotron...", fallback="gemma...")
        logger.pipeline_start(n=5, total_batches=3)
        logger.pipeline_complete(total=5)
        logger.pipeline_partial(collected=3, requested=5)
        logger.save(file_path="output/cinematic_indian_tantra_music.json")
        logger.section(title="OUTPUT")
    """

    def __init__(self) -> None:
        self._console = console

    # -------------------------------------------------------------------------
    # Pipeline Level
    # -------------------------------------------------------------------------

    def pipeline_start(self, n: int, total_batches: int) -> None:
        """
        Log pipeline initialization.

        Args:
            n             (int): Total variations requested.
            total_batches (int): Total batches to run.
        """
        self._console.print(
            _format_event(
                symbol  = SYMBOLS["info"],
                style   = "info",
                label   = "PIPELINE",
                message = f"Starting — ",
                detail  = f"{n} variations across {total_batches} batches",
            )
        )

    def pipeline_complete(self, total: int) -> None:
        """
        Log successful pipeline completion.

        Args:
            total (int): Total variations collected.
        """
        self._console.print(
            _format_event(
                symbol  = SYMBOLS["success"],
                style   = "success",
                label   = "PIPELINE",
                message = f"Complete",
                detail  = f"{total} variations generated successfully",
            )
        )

    def pipeline_partial(self, collected: int, requested: int) -> None:
        """
        Log partial pipeline completion — some batches failed.

        Args:
            collected (int): Variations successfully collected.
            requested (int): Total variations originally requested.
        """
        self._console.print(
            _format_event(
                symbol  = SYMBOLS["warning"],
                style   = "warning",
                label   = "PIPELINE",
                message = f"Partial result",
                detail  = f"{collected}/{requested} variations collected",
            )
        )

    def pipeline_failed(self, reason: str) -> None:
        """
        Log full pipeline failure.

        Args:
            reason (str): Short failure reason.
        """
        self._console.print(
            _format_event(
                symbol  = SYMBOLS["error"],
                style   = "error",
                label   = "PIPELINE",
                message = f"Failed",
                detail  = reason,
            )
        )

    # -------------------------------------------------------------------------
    # Batch Level
    # -------------------------------------------------------------------------

    def batch_start(
        self,
        batch_num:   int,
        total:       int,
        size:        int,
        start_index: int,
    ) -> None:
        """
        Log the start of a batch request.

        Args:
            batch_num   (int): Current batch number (1-based).
            total       (int): Total batches in pipeline.
            size        (int): Variations being requested in this batch.
            start_index (int): Variation index this batch starts from.
        """
        self._console.print(
            _format_event(
                symbol  = SYMBOLS["batch"],
                style   = "batch",
                label   = f"BATCH {batch_num}/{total}",
                message = f"Requesting {size} variation(s)",
                detail  = f"index start: {start_index}",
            )
        )

    def batch_success(
        self,
        batch_num: int,
        received:  int,
        model:     str,
    ) -> None:
        """
        Log a successful batch response.

        Args:
            batch_num (int): Batch number that succeeded.
            received  (int): Number of variations received.
            model     (str): Model that responded.
        """
        short_model = model.split("/")[-1]
        self._console.print(
            _format_event(
                symbol  = SYMBOLS["success"],
                style   = "success",
                label   = f"BATCH {batch_num}",
                message = f"{received} variation(s) received",
                detail  = f"via {short_model}",
            )
        )

    def batch_failed(self, batch_num: int, reason: str) -> None:
        """
        Log a batch that failed after all retries and fallback.

        Args:
            batch_num (int): Batch number that failed.
            reason    (str): Short error description.
        """
        self._console.print(
            _format_event(
                symbol  = SYMBOLS["error"],
                style   = "error",
                label   = f"BATCH {batch_num}",
                message = f"Failed after all retries",
                detail  = reason,
            )
        )

    # -------------------------------------------------------------------------
    # Model Level
    # -------------------------------------------------------------------------

    def attempt(self, attempt: int, total: int, model: str) -> None:
        """
        Log a single generation attempt.

        Args:
            attempt (int): Current attempt number.
            total   (int): Max attempts allowed.
            model   (str): Model being attempted.
        """
        short_model = model.split("/")[-1]
        self._console.print(
            _format_event(
                symbol  = SYMBOLS["spin"],
                style   = "info",
                label   = f"ATTEMPT {attempt}/{total}",
                message = f"Requesting",
                detail  = short_model,
            )
        )

    def retry(self, attempt: int, total: int, model: str, error: str, wait: int) -> None:
        """
        Log a retry after a failed attempt.

        Args:
            attempt (int): Attempt number that just failed.
            total   (int): Max attempts allowed.
            model   (str): Model that failed.
            error   (str): Short error message.
            wait    (int): Seconds before next retry.
        """
        short_model = model.split("/")[-1]
        self._console.print(
            _format_event(
                symbol  = SYMBOLS["warning"],
                style   = "warning",
                label   = f"RETRY {attempt}/{total}",
                message = f"{error}",
                detail  = f"{short_model} · retrying in {wait}s",
            )
        )

    def model_fallback(self, primary: str, fallback: str) -> None:
        """
        Log the switch from primary to fallback model.

        Args:
            primary  (str): Primary model that exhausted retries.
            fallback (str): Fallback model being switched to.
        """
        short_primary  = primary.split("/")[-1]
        short_fallback = fallback.split("/")[-1]
        self._console.print(
            _format_event(
                symbol  = SYMBOLS["arrow"],
                style   = "warning",
                label   = "FALLBACK",
                message = f"{short_primary}",
                detail  = f"switching to {short_fallback}",
            )
        )

    def model_success(self, model: str, attempt: int) -> None:
        """
        Log a successful model response.

        Args:
            model   (str): Model that responded.
            attempt (int): Attempt number it succeeded on.
        """
        short_model = model.split("/")[-1]
        self._console.print(
            _format_event(
                symbol  = SYMBOLS["success"],
                style   = "success",
                label   = "MODEL",
                message = f"Responded on attempt {attempt}",
                detail  = short_model,
            )
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    def save(self, file_path: str) -> None:
        """
        Log a successful file save.

        Args:
            file_path (str): Path where the file was saved.
        """
        self._console.print(
            _format_event(
                symbol  = SYMBOLS["success"],
                style   = "success",
                label   = "SAVED",
                message = f"{file_path}",
            )
        )

    # -------------------------------------------------------------------------
    # Section Dividers
    # -------------------------------------------------------------------------

    def section(self, title: str) -> None:
        """
        Print a labeled section divider Rule.

        Args:
            title (str): Section title displayed in the center of the rule.
        """
        self._console.print()
        self._console.print(
            Rule(
                title        = f"[muted] {title} [/]",
                style        = COLORS["muted"],
                align        = "center",
            )
        )
        self._console.print()

    def blank(self) -> None:
        """Print an empty line — for breathing room between log groups."""
        self._console.print()
        