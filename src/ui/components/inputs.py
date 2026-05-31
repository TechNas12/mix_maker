# src/ui/components/inputs.py

from rich.prompt    import Prompt, IntPrompt
from rich.table     import Table
from rich.panel     import Panel
from rich.text      import Text
from rich.rule      import Rule
from src.ui.theme   import console, COLORS, BORDER_STYLE, PANEL_PADDING, SYMBOLS
from src.utils.load_json import load_json
from src.utils.exception import CustomException


# -------------------------------------------------------------------------
# Internal Helpers
# -------------------------------------------------------------------------

def _load_styles(style_prompts_file: str) -> list[dict]:
    """
    Load and return all styles from the style_prompts.json file.

    Args:
        style_prompts_file (str): Path to style_prompts.json.

    Returns:
        list[dict]: List of style dicts with id, label, description.

    Raises:
        CustomException: If the file cannot be loaded or is empty.
    """
    styles = load_json(style_prompts_file)

    if not styles:
        raise CustomException(
            error   = ValueError("Style prompts file is empty."),
            message = "No styles found in style_prompts.json.",
            context = {"file": style_prompts_file}
        )

    return styles


def _print_style_table(styles: list[dict]) -> None:
    """
    Print a numbered style selection table to the terminal.

    Each row shows: index, style label, short description.

    Args:
        styles (list[dict]): List of style dicts loaded from style_prompts.json.
    """
    table = Table(
        show_header  = True,
        header_style = f"bold {COLORS['primary']}",
        box          = None,
        padding      = (0, 2),
        show_edge    = False,
        expand       = True,
    )

    table.add_column("#",           style=f"bold {COLORS['secondary']}", ratio=1, justify="right")
    table.add_column("STYLE",       style=f"bold {COLORS['primary']}",   ratio=2)
    table.add_column("DESCRIPTION", style="text",                         ratio=6)

    for i, style in enumerate(styles, start=1):
        table.add_row(
            str(i),
            style.get("label", style["id"]).upper(),
            style.get("description", "")[:80] + "..."
            if len(style.get("description", "")) > 80
            else style.get("description", ""),
        )

    console.print(
        Panel(
            table,
            title        = f"[primary] {SYMBOLS['info']} AVAILABLE STYLES [/]",
            border_style = BORDER_STYLE,
            padding      = PANEL_PADDING,
        )
    )
    console.print()


def _print_summary(user_prompt: str, style: dict, n: int) -> None:
    """
    Print a confirmation summary of collected inputs before the pipeline runs.

    Args:
        user_prompt (str):  The user's music prompt.
        style       (dict): The selected style dict.
        n           (int):  Number of variations requested.
    """
    table = Table(
        show_header = False,
        box         = None,
        padding     = (0, 2),
        show_edge   = False,
        expand      = True,
    )

    table.add_column("key",   style=f"bold {COLORS['muted']}",   ratio=2)
    table.add_column("value", style=f"bold {COLORS['primary']}", ratio=5)

    table.add_row(
        f"{SYMBOLS['info']} PROMPT",
        user_prompt,
    )
    table.add_row(
        f"{SYMBOLS['batch']} STYLE",
        f"{style.get('label', style['id']).upper()}  "
        f"[muted]({style['id']})[/]",
    )
    table.add_row(
        f"{SYMBOLS['info']} VARIATIONS",
        str(n),
    )

    console.print(
        Panel(
            table,
            title        = f"[primary] {SYMBOLS['info']} RUN CONFIGURATION [/]",
            border_style = BORDER_STYLE,
            padding      = PANEL_PADDING,
        )
    )
    console.print()


# -------------------------------------------------------------------------
# Public
# -------------------------------------------------------------------------

def collect_inputs(style_prompts_file: str) -> tuple[str, str, int]:
    """
    Interactively collect pipeline inputs from the user via styled Rich prompts.

    Renders in order:
        1. Section rule
        2. Music prompt input
        3. Style table + numbered selection
        4. Variation count input
        5. Summary confirmation panel
        6. Confirm or restart loop

    Args:
        style_prompts_file (str): Path to style_prompts.json —
                                  used to dynamically populate style choices.

    Returns:
        tuple[str, str, int]: (user_prompt, style_id, n)
            user_prompt (str): The music description entered by the user.
            style_id    (str): The selected style's id string.
            n           (int): Number of variations to generate.

    Raises:
        CustomException: If styles cannot be loaded.
    """
    styles = _load_styles(style_prompts_file)

    while True:
        # ── Section header ────────────────────────────────────────────────
        console.print()
        console.print(
            Rule(
                title  = f"[primary] {SYMBOLS['info']} INPUT CONFIGURATION [/]",
                style  = COLORS["primary"],
                align  = "center",
            )
        )
        console.print()

        # ── Prompt input ──────────────────────────────────────────────────
        user_prompt = Prompt.ask(
            f"  [bold {COLORS['primary']}]{SYMBOLS['info']}[/] "
            f"[bold]Enter your music prompt[/]",
            console = console,
        ).strip()

        if not user_prompt:
            console.print(
                f"  [error]{SYMBOLS['error']}  Prompt cannot be empty. "
                f"Please try again.[/]"
            )
            continue

        console.print()

        # ── Style selection ───────────────────────────────────────────────
        _print_style_table(styles)

        while True:
            style_choice = IntPrompt.ask(
                f"  [bold {COLORS['primary']}]{SYMBOLS['batch']}[/] "
                f"[bold]Select a style[/] "
                f"[muted](1–{len(styles)})[/]",
                console = console,
            )

            if 1 <= style_choice <= len(styles):
                selected_style = styles[style_choice - 1]
                break

            console.print(
                f"  [error]{SYMBOLS['error']}  Invalid choice. "
                f"Enter a number between 1 and {len(styles)}.[/]"
            )

        console.print()

        # ── Variation count ───────────────────────────────────────────────
        while True:
            n = IntPrompt.ask(
                f"  [bold {COLORS['primary']}]{SYMBOLS['info']}[/] "
                f"[bold]Number of variations[/] "
                f"[muted](1–20)[/]",
                default = 5,
                console = console,
            )

            if 1 <= n <= 20:
                break

            console.print(
                f"  [error]{SYMBOLS['error']}  "
                f"Please enter a number between 1 and 20.[/]"
            )

        console.print()

        # ── Summary confirmation ──────────────────────────────────────────
        _print_summary(
            user_prompt = user_prompt,
            style       = selected_style,
            n           = n,
        )

        confirm = Prompt.ask(
            f"  [bold {COLORS['secondary']}]{SYMBOLS['arrow']}[/] "
            f"[bold]Confirm and run?[/] "
            f"[muted](yes / no to re-enter)[/]",
            choices = ["yes", "no"],
            default = "yes",
            console = console,
        )

        if confirm == "yes":
            console.print()
            return user_prompt, selected_style["id"], n

        # Loop back if user wants to re-enter
        console.print(
            f"\n  [muted]{SYMBOLS['info']}  Restarting input...[/]\n"
        )