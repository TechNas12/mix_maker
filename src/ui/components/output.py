# src/ui/components/output.py

from rich.panel     import Panel
from rich.text      import Text
from rich.rule      import Rule
from rich.columns   import Columns
from rich.syntax    import Syntax
from rich.table     import Table
import json
from src.ui.theme   import console, COLORS, BORDER_STYLE, PANEL_PADDING, SYMBOLS


# -------------------------------------------------------------------------
# Internal Helpers
# -------------------------------------------------------------------------

def _build_meta_table(meta: dict) -> Table:
    """
    Build a styled summary table from the pipeline meta block.

    Displays: total requested, total returned, batches completed,
    and whether the run was complete or partial.

    Args:
        meta (dict): The 'meta' block from generate_response() output.

    Returns:
        Table: Rich Table ready to print.
    """
    table = Table(
        show_header  = False,
        box          = None,
        padding      = (0, 2),
        show_edge    = False,
        expand       = True,
    )

    table.add_column("key",   style=f"bold {COLORS['muted']}",    ratio=1)
    table.add_column("value", style=f"bold {COLORS['primary']}",  ratio=1)
    table.add_column("key2",  style=f"bold {COLORS['muted']}",    ratio=1)
    table.add_column("value2",style=f"bold {COLORS['secondary']}", ratio=1)

    requested = meta.get("total_variations_requested", "—")
    returned  = meta.get("total_variations_returned",  "—")
    batches   = meta.get("batches_completed", "—")
    total_b   = meta.get("total_batches", "—")
    complete  = meta.get("complete", False)

    complete_text = (
        f"[success]{SYMBOLS['success']} YES[/]"
        if complete
        else f"[warning]{SYMBOLS['warning']} PARTIAL[/]"
    )

    table.add_row(
        f"{SYMBOLS['info']} REQUESTED",  str(requested),
        f"{SYMBOLS['info']} RETURNED",   str(returned),
    )
    table.add_row(
        f"{SYMBOLS['batch']} BATCHES",   f"{batches}/{total_b}",
        f"{SYMBOLS['info']} COMPLETE",   complete_text,
    )

    return table


def _build_variation_panel(variation: dict, total: int) -> Panel:
    """
    Build a single variation panel.

    Each variation gets its own bordered panel with:
        - Index badge in the title
        - Prompt text as the body

    Args:
        variation (dict): Single variation dict {"index": int, "prompt": str}.
        total     (int):  Total variations — used in the title for context.

    Returns:
        Panel: Rich Panel for this variation.
    """
    index  = variation.get("index", "?")
    prompt = variation.get("prompt", "")

    body = Text(prompt, style="text", justify="full")

    return Panel(
        body,
        title        = (
            f"[secondary] {SYMBOLS['batch']} VARIATION {index} [/]"
            f"[muted] of {total} [/]"
        ),
        border_style = COLORS["secondary"],
        padding      = PANEL_PADDING,
    )


# -------------------------------------------------------------------------
# Public
# -------------------------------------------------------------------------

def print_output(result: dict) -> None:
    """
    Render the full pipeline output to the terminal.

    Renders in order:
        1. Section header rule
        2. Run summary meta table (inside a panel)
        3. Each variation in its own panel, sequentially
        4. Closing rule

    Args:
        result (dict): The dict returned by LLM.generate_response() containing
                       'style_id', 'base_prompt', 'complete', 'variations', 'meta'.
    """
    style_id    = result.get("style_id",    "unknown")
    base_prompt = result.get("base_prompt", "")
    variations  = result.get("variations",  [])
    meta        = result.get("meta",        {})
    complete    = result.get("complete",    False)

    # ── Section header ────────────────────────────────────────────────────
    console.print()
    console.print(
        Rule(
            title  = f"[primary] {SYMBOLS['info']} OUTPUT [/]",
            style  = COLORS["primary"],
            align  = "center",
        )
    )
    console.print()

    # ── Run summary ───────────────────────────────────────────────────────
    meta_table = _build_meta_table({**meta, "complete": complete})

    console.print(
        Panel(
            meta_table,
            title        = (
                f"[primary] {SYMBOLS['info']} {style_id.upper()} [/]"
                f"[muted] {SYMBOLS['arrow']} {base_prompt} [/]"
            ),
            border_style = BORDER_STYLE,
            padding      = PANEL_PADDING,
        )
    )
    console.print()

    # ── Variations ────────────────────────────────────────────────────────
    total = len(variations)

    if not variations:
        console.print(
            Panel(
                Text(
                    f"  {SYMBOLS['error']}  No variations were generated.",
                    style="error",
                    justify="center",
                ),
                border_style = COLORS["error"],
                padding      = PANEL_PADDING,
            )
        )
        return

    for variation in variations:
        console.print(_build_variation_panel(variation, total))
        console.print()

    # ── Closing rule ──────────────────────────────────────────────────────
    status_style  = "success" if complete else "warning"
    status_symbol = SYMBOLS["success"] if complete else SYMBOLS["warning"]
    status_text   = "COMPLETE" if complete else "PARTIAL"

    console.print(
        Rule(
            title  = f"[{status_style}] {status_symbol} {status_text} [/]",
            style  = COLORS[status_style],
            align  = "center",
        )
    )
    console.print()


def print_save_confirmation(file_path: str) -> None:
    """
    Print a small confirmation panel after the output is saved to disk.

    Args:
        file_path (str): Path where the JSON was saved.
    """
    body = Text(justify="center")
    body.append(f"\n  {SYMBOLS['success']}  Saved to  ", style="muted")
    body.append(f"{file_path}",                          style=f"bold {COLORS['primary']}")
    body.append("\n")

    console.print(
        Panel(
            body,
            border_style = COLORS["success"],
            padding      = (0, 2),
        )
    )
    console.print()