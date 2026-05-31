# src/ui/theme.py

from rich.console import Console
from rich.theme import Theme

# -------------------------------------------------------------------------
# Color Palette
# -------------------------------------------------------------------------

COLORS = {
    "primary":      "#00F5FF",   # cyan — main accent
    "secondary":    "#BD00FF",   # purple — secondary accent  
    "success":      "#00FF94",   # green — success states
    "warning":      "#FFD600",   # yellow — warnings
    "error":        "#FF4444",   # red — errors / failures
    "critical":     "#FF0000",   # bright red — critical failures
    "muted":        "#555555",   # dim text, timestamps
    "text":         "#E0E0E0",   # body text
    "border":       "#00F5FF",   # panel borders
}

# -------------------------------------------------------------------------
# Rich Theme — maps semantic names to styles
# used as [tag] markup throughout all components
# -------------------------------------------------------------------------

APP_THEME = Theme({
    "primary":      f"bold {COLORS['primary']}",
    "secondary":    f"bold {COLORS['secondary']}",
    "success":      f"bold {COLORS['success']}",
    "warning":      f"bold {COLORS['warning']}",
    "error":        f"bold {COLORS['error']}",
    "critical":     f"bold {COLORS['critical']}",
    "muted":        COLORS["muted"],
    "text":         COLORS["text"],
    "info":         f"{COLORS['primary']}",

    # Pipeline-specific semantic tags
    "batch":        f"bold {COLORS['secondary']}",
    "model":        f"italic {COLORS['primary']}",
    "step":         f"{COLORS['text']}",
    "highlight":    f"bold {COLORS['secondary']}",
})

# -------------------------------------------------------------------------
# Shared Console Instance
# Every component imports THIS — never creates its own Console()
# -------------------------------------------------------------------------

console = Console(theme=APP_THEME)

# -------------------------------------------------------------------------
# Panel / Border Style
# -------------------------------------------------------------------------

BORDER_STYLE    = COLORS["border"]
PANEL_PADDING   = (1, 2)          # (vertical, horizontal)

# -------------------------------------------------------------------------
# Symbols — consistent iconography across all components
# -------------------------------------------------------------------------

SYMBOLS = {
    "success":  "✔",
    "error":    "✘",
    "warning":  "⚠",
    "info":     "✦",
    "arrow":    "→",
    "batch":    "⬡",
    "model":    "⬢",
    "spin":     "⠿",
    "divider":  "─",
}