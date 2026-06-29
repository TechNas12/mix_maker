# src/ui/components/music_panel.py

import time
import threading
from rich.live      import Live
from rich.table     import Table
from rich.panel     import Panel
from rich.text      import Text
from rich.spinner   import Spinner
from rich.progress  import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.console   import Group
from src.ui.theme   import console, COLORS, BORDER_STYLE, PANEL_PADDING, SYMBOLS


# -------------------------------------------------------------------------
# Music Generation Live Panel
# -------------------------------------------------------------------------

class MusicPanel:
    """
    Live terminal panel for concurrent audio generation.

    Shows one row per variation with a per-row status that updates
    as each variation completes via the on_complete callback.

    Each row transitions:
        ·  Queued  →  ⠿ Generating...  →  ✔ file.mp3  2.4MB
                                        →  ✘ error msg

    Usage:
        panel = MusicPanel(variations=result["variations"])

        with panel:
            music_result = generator.generate(
                llm_result  = result,
                on_complete = panel.on_variation_complete,
            )

        panel.add_stitch_result(music_result)
    """

    # Row state constants
    _QUEUED     = "queued"
    _GENERATING = "generating"
    _SUCCESS    = "success"
    _FAILED     = "failed"

    def __init__(self, variations: list[dict]) -> None:
        """
        Args:
            variations (list[dict]): Variation list from LLM result.
        """
        self._variations  = variations
        self._total       = len(variations)
        self._completed   = 0
        self._lock        = threading.Lock()
        self._live        = None
        self._start_time  = None
        self._stitch      = None   # populated after stitching

        # Per-row state — keyed by variation index
        self._rows: dict[int, dict] = {
            v["index"]: {
                "state":    self._QUEUED,
                "file":     None,
                "size_mb":  None,
                "error":    None,
            }
            for v in variations
        }

        # Overall progress spinner
        self._progress = Progress(
            SpinnerColumn(
                spinner_name   = "dots2",
                style          = f"bold {COLORS['primary']}",
                finished_text  = f"[success]{SYMBOLS['success']}[/]",
            ),
            TextColumn("[step]{task.description}[/]"),
            MofNCompleteColumn(separator=" of "),
            TimeElapsedColumn(),
            console          = console,
            transient        = False,
        )
        self._task_id = self._progress.add_task(
            description = "  Generating audio...",
            total       = self._total,
        )

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> "MusicPanel":
        self._start_time = time.time()

        # Mark all rows as generating at start
        # (they're all fired concurrently so all start immediately)
        with self._lock:
            for index in self._rows:
                self._rows[index]["state"] = self._GENERATING

        self._live = Live(
            self._render(),
            console           = console,
            refresh_per_second= 12,
            transient         = False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._live:
            self._live.update(self._render())
            self._live.__exit__(exc_type, exc_val, exc_tb)

    # -------------------------------------------------------------------------
    # Callback — called by MusicGenerator._generate_single on completion
    # -------------------------------------------------------------------------

    def on_variation_complete(self, result: dict) -> None:
        """
        Called by MusicGenerator when a single variation finishes.
        Thread-safe — asyncio callbacks may fire from different threads.

        Args:
            result (dict): Result dict from _generate_single.
        """
        index  = result.get("index")
        status = result.get("status")

        with self._lock:
            self._rows[index]["state"]   = (
                self._SUCCESS if status == "success" else self._FAILED
            )
            self._rows[index]["file"]    = result.get("file")
            self._rows[index]["size_mb"] = result.get("size_mb")
            self._rows[index]["error"]   = result.get("error")
            self._completed += 1

        # Advance overall progress
        self._progress.update(
            self._task_id,
            advance     = 1,
            description = (
                f"  Generating audio..."
                if self._completed < self._total
                else f"  [success]Audio generation complete[/]"
            ),
        )

        # Refresh live display
        if self._live:
            self._live.update(self._render())

    # -------------------------------------------------------------------------
    # Stitch result — called after MixMaker.stitch() completes
    # -------------------------------------------------------------------------

    def add_stitch_result(self, music_result: dict) -> None:
        """
        Add the stitch result block to the panel and do a final render.

        Args:
            music_result (dict): Updated manifest from MixMaker.stitch().
        """
        self._stitch = music_result.get("stitched")
        if self._live:
            self._live.update(self._render())

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def _render_rows(self) -> Table:
        """Build the per-variation rows table."""
        table = Table(
            show_header = False,
            box         = None,
            padding     = (0, 1),
            show_edge   = False,
            expand      = True,
        )

        table.add_column("idx",    style=f"bold {COLORS['muted']}",    width=4,  justify="right")
        table.add_column("status", width=4,                             justify="center")
        table.add_column("detail", style="text",                        ratio=1)

        for v in self._variations:
            index = v["index"]
            row   = self._rows[index]
            state = row["state"]

            if state == self._QUEUED:
                status_cell = Text("·", style=f"dim {COLORS['muted']}")
                detail_cell = Text("Queued", style=f"dim {COLORS['muted']}")

            elif state == self._GENERATING:
                spinner     = Spinner("dots2", style=f"bold {COLORS['primary']}")
                status_cell = spinner
                detail_cell = Text("Generating...", style=f"italic {COLORS['primary']}")

            elif state == self._SUCCESS:
                status_cell = Text(SYMBOLS["success"], style="success")
                size_str    = (
                    f"  {row['size_mb']}MB"
                    if row["size_mb"] else ""
                )
                detail_cell = Text()
                detail_cell.append(row["file"] or "",  style=f"bold {COLORS['primary']}")
                detail_cell.append(size_str,           style=f"dim {COLORS['muted']}")

            else:  # failed
                status_cell = Text(SYMBOLS["error"], style="error")
                detail_cell = Text(
                    row["error"] or "Unknown error",
                    style=f"bold {COLORS['error']}"
                )

            table.add_row(str(index), status_cell, detail_cell)

        return table

    def _render_stitch(self) -> Text | None:
        """Build the stitch result line if available."""
        if not self._stitch:
            return None

        mode     = self._stitch.get("mode", "—")
        included = self._stitch.get("tracks_included", 0)
        skipped  = self._stitch.get("tracks_skipped", 0)
        path     = self._stitch.get("file_path", "—")

        line = Text()
        line.append(f"\n  {SYMBOLS['success']}  ", style="success")
        line.append("Stitched ",                    style=f"bold {COLORS['success']}")
        line.append(f"{SYMBOLS['arrow']} ",         style="muted")
        line.append(f"{path}",                      style=f"bold {COLORS['primary']}")
        line.append(
            f"\n     Mode: {mode}  ·  "
            f"{included} tracks  ·  "
            f"{skipped} skipped",
            style=f"dim {COLORS['muted']}"
        )

        return line

    def _render(self) -> Panel:
        """Assemble the full live panel."""
        rows        = self._render_rows()
        stitch_line = self._render_stitch()

        # Stack: rows → blank → progress spinner → optional stitch
        renderables = [rows, Text(""), self._progress]

        if stitch_line:
            renderables.append(stitch_line)

        return Panel(
            Group(*renderables),
            title        = f"[primary] {SYMBOLS['batch']} AUDIO GENERATION [/]",
            border_style = BORDER_STYLE,
            padding      = PANEL_PADDING,
        )