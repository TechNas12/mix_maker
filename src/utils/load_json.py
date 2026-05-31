import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
console = Console()

from src.utils.logger import log
from src.utils.exception import CustomException

def save_json(
    data: Any,
    file_name: str,
    directory: str | Path = "output",
    indent: int = 2,
    overwrite: bool = True,
) -> Path:
    """
    Save a Python object as a JSON file in a relative directory.
 
    Args:
        data:       The Python object to serialize (dict, list, etc.)
        file_name:  File name — with or without .json extension.
        directory:  Relative directory path. Created if it doesn't exist. Default: "output"
        indent:     JSON indentation level. Default: 2
        overwrite:  If False, raises an exception when the file already exists. Default: True
 
    Returns:
        Path: The resolved path of the saved file.
 
    Raises:
        CustomException: If serialization fails, directory cannot be created,
                         file already exists (when overwrite=False), or write fails.
    """
    try:
        # Ensure .json extension
        if not file_name.endswith(".json"):
            file_name = f"{file_name}.json"
 
        # Resolve and create directory
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
 
        file_path = dir_path / file_name
 
        # Guard against accidental overwrites
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {file_path}")
 
        # Serialize — do this before opening the file so a bad object
        # doesn't produce an empty/corrupt file on disk
        serialized = json.dumps(data, indent=indent, ensure_ascii=False)
 
        file_path.write_text(serialized, encoding="utf-8")
 
        log(f"JSON saved → {file_path} ({len(serialized)} bytes)", level="info")
        return file_path
 
    except FileExistsError as e:
        error_msg = f"File already exists and overwrite=False: {file_path}"
        log(error_msg, level="error")
        raise CustomException(
            error=e,
            message=error_msg,
            context={"file_path": str(file_path), "error_type": "FileExistsError"},
        )
 
    except TypeError as e:
        error_msg = f"Object is not JSON-serializable: {type(data).__name__}"
        log(error_msg, level="error")
        raise CustomException(
            error=e,
            message=error_msg,
            context={
                "data_type": type(data).__name__,
                "error_type": "TypeError",
            },
        )
 
    except OSError as e:
        error_msg = f"Failed to write file '{file_path}': {e.strerror}"
        log(error_msg, level="error")
        raise CustomException(
            error=e,
            message=error_msg,
            context={"file_path": str(file_path), "error_type": "OSError"},
        )
 
    except Exception as e:
        error_msg = f"Unexpected error saving JSON to '{file_path}': {type(e).__name__}"
        log(error_msg, level="critical")
        raise CustomException(
            error=e,
            message=error_msg,
            context={
                "file_path": str(file_path),
                "error_type": type(e).__name__,
            },
        )
        

def display_json(
    data: Any,
    title: str | None = None,
    indent: int = 2,
    ) -> None:
    """
    Pretty-print a Python object as syntax-highlighted JSON in the terminal.
 
    Uses Rich for colored output. Falls back to plain print if Rich fails.
 
    Args:
        data:   The Python object to display (dict, list, dataclass, etc.)
        title:  Optional panel title shown above the JSON block.
        indent: JSON indentation level. Default: 2
 
    Raises:
        CustomException: If the object cannot be serialized to JSON.
    """
    try:
        serialized = json.dumps(data, indent=indent, ensure_ascii=False)
 
        syntax = Syntax(
            serialized,
            "json",
            theme="monokai",
            line_numbers=False,
            word_wrap=True,
        )
 
        if title:
            console.print(Panel(syntax, title=f"[bold cyan]{title}[/bold cyan]", border_style="dim"))
        else:
            console.print(syntax)
 
        log(f"Displayed JSON | title: '{title}' | {len(serialized)} chars", level="info")
 
    except TypeError as e:
        error_msg = f"Object is not JSON-serializable: {type(data).__name__}"
        log(error_msg, level="error")
        raise CustomException(
            error=e,
            message=error_msg,
            context={
                "data_type": type(data).__name__,
                "error_type": "TypeError",
            },
        )
 
    except Exception as e:
        # Rich itself failed — fall back to plain output so display never hard-crashes
        log(f"Rich display failed, falling back to plain print: {e}", level="warning")
        print(json.dumps(data, indent=indent, ensure_ascii=False))


def load_json(
    file_path: str | Path, 
    allow_empty: bool = False,
    max_size_mb: int = 100
) -> Any:
    """
    Load a JSON file and return its contents as a Python object.

    Args:
        file_path: Path to the JSON file
        allow_empty: If False, raises exception for empty files (default: False)
        max_size_mb: Maximum file size in MB to prevent memory issues (default: 100)

    Returns:
        dict, list, str, int, float, bool, or None - the parsed JSON content

    Raises:
        CustomException: If file cannot be read, JSON is invalid, or validation fails
    """

    try:
        file_path = Path(file_path)

        # ✅ Check file size before loading (prevent memory issues)
        if file_path.stat().st_size == 0:
            if allow_empty:
                log(f"JSON file is empty: {file_path}", level="warning")
                return None
            else:
                raise ValueError("JSON file is empty")

        if file_path.stat().st_size > max_size_mb * 1024 * 1024:
            raise ValueError(
                f"JSON file exceeds maximum size of {max_size_mb}MB"
            )

        # ✅ Load and parse JSON
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        log(f"Successfully loaded JSON file: {file_path}", level="info")
        return data

    # ✅ Handle specific exceptions with clear messages
    except FileNotFoundError as e:
        error_msg = f"JSON file not found: {file_path}"
        log(error_msg, level="error")
        raise CustomException(
            error=e,
            message=error_msg,
            context={"file_path": str(file_path), "error_type": "FileNotFoundError"}
        )

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON syntax in file '{file_path}' at line {e.lineno}, column {e.colno}: {e.msg}"
        log(error_msg, level="error")
        raise CustomException(
            error=e,
            message=error_msg,
            context={
                "file_path": str(file_path),
                "error_type": "JSONDecodeError",
                "line": e.lineno,
                "column": e.colno
            }
        )

    except UnicodeDecodeError as e:
        error_msg = f"File encoding error in '{file_path}': {e.reason}"
        log(error_msg, level="error")
        raise CustomException(
            error=e,
            message=error_msg,
            context={
                "file_path": str(file_path),
                "error_type": "UnicodeDecodeError"
            }
        )

    except ValueError as e:
        error_msg = f"Invalid file: {str(e)}"
        log(error_msg, level="error")
        raise CustomException(
            error=e,
            message=error_msg,
            context={"file_path": str(file_path), "error_type": "ValueError"}
        )

    except Exception as e:
        # ✅ Last resort - catch unexpected errors with logging
        error_msg = f"Unexpected error loading JSON file '{file_path}': {type(e).__name__}"
        log(error_msg, level="critical")
        raise CustomException(
            error=e,
            message=error_msg,
            context={
                "file_path": str(file_path),
                "error_type": type(e).__name__
            }
        )