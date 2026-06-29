# src/ui/components/header.py

from rich.panel   import Panel
from rich.text    import Text
from rich.columns import Columns
from rich.rule    import Rule
from src.ui.theme import console, COLORS, BORDER_STYLE, PANEL_PADDING, SYMBOLS

# -------------------------------------------------------------------------
# ASCII Banner
# -------------------------------------------------------------------------

BANNER = """
███╗   ███╗██╗██╗  ██╗    ███╗   ███╗ █████╗ ██╗  ██╗███████╗██████╗ 
████╗ ████║██║╚██╗██╔╝    ████╗ ████║██╔══██╗██║ ██╔╝██╔════╝██╔══██╗
██╔████╔██║██║ ╚███╔╝     ██╔████╔██║███████║█████╔╝ █████╗  ██████╔╝
██║╚██╔╝██║██║ ██╔██╗     ██║╚██╔╝██║██╔══██║██╔═██╗ ██╔══╝  ██╔══██╗
██║ ╚═╝ ██║██║██╔╝ ██╗    ██║ ╚═╝ ██║██║  ██║██║  ██╗███████╗██║  ██║
╚═╝     ╚═╝╚═╝╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
"""

# -------------------------------------------------------------------------
# Version & Meta
# -------------------------------------------------------------------------

APP_VERSION  = "created by TechNas"
APP_SUBTITLE = "AI-Powered Music Generator"

# -------------------------------------------------------------------------
# Internal Builders
# -------------------------------------------------------------------------

def _build_banner() -> Text:
    """Styled ASCII banner text."""
    text = Text(BANNER, style=f"bold {COLORS['primary']}", justify="center")
    return text


def _build_subtitle() -> Text:
    """Subtitle line below the banner."""
    text = Text(justify="center")
    text.append(f"  {APP_SUBTITLE}  ", style=f"bold {COLORS['secondary']}")
    text.append(f" {APP_VERSION} ", style=f"dim {COLORS['muted']}")
    return text


def _build_info_grid(
    primary_model: str,
    fallback_model: str,
    style_id: str,
    user_prompt: str,
    n: int,
) -> Columns:
    """
    Two-column info grid showing run configuration.

    Left  → model info
    Right → generation params
    """
    left = Text()
    left.append(f" {SYMBOLS['model']} PRIMARY    ", style="muted")
    left.append(f"{primary_model}\n",               style="model")
    left.append(f" {SYMBOLS['model']} FALLBACK   ", style="muted")
    left.append(f"{fallback_model}",                style=f"dim {COLORS['secondary']}")

    right = Text()
    right.append(f" {SYMBOLS['info']} STYLE      ", style="muted")
    right.append(f"{style_id.upper()}\n",           style="highlight")
    right.append(f" {SYMBOLS['info']} VARIATIONS ", style="muted")
    right.append(f"{n}",                            style="highlight")
    right.append(f"   {SYMBOLS['arrow']} ",         style="muted")
    right.append(f"{user_prompt}",                  style="text")

    return Columns([left, right], equal=True, expand=True)


# -------------------------------------------------------------------------
# Public
# -------------------------------------------------------------------------

def print_header(
    primary_model: str,
    fallback_model: str,
    style_id: str,
    user_prompt: str,
    n: int,
) -> None:
    """
    Print the full application header to the terminal.

    Renders: ASCII banner → subtitle → divider → run configuration grid.
    Called once at the start of interface.py before the pipeline runs.

    Args:
        primary_model  (str): Primary model identifier string.
        fallback_model (str): Fallback model identifier string.
        style_id       (str): Style ID being used for this run.
        user_prompt    (str): The user's input prompt.
        n              (int): Number of variations requested.
    """
    # Banner
    console.print(_build_banner())
    console.print(_build_subtitle())
    console.print()

    # Run config inside a panel
    grid = _build_info_grid(primary_model, fallback_model, style_id, user_prompt, n)
    console.print(
        Panel(
            grid,
            title=f"[primary] {SYMBOLS['info']} RUN CONFIGURATION [/]",
            border_style=BORDER_STYLE,
            padding=PANEL_PADDING,
        )
    )

    # Closing divider before pipeline starts
    console.print(
        Rule(
            title=f"[muted] PIPELINE START [/]",
            style=COLORS["muted"],
            align="center",
        )
    )
    console.print()
    
# src/ui/components/header.py — add this function

def print_banner_only() -> None:
    """Print just the ASCII banner and subtitle — before inputs are collected."""
    console.print(_build_banner())
    console.print(_build_subtitle())
    console.print()