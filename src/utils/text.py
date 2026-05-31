from pathlib import Path


def load_text_file(file_path: str) -> str:
    """
    Load and return the contents of a text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: File contents.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return path.read_text(encoding="utf-8")


def export_to_text(content: str, filename: str) -> str:
    """
    Save content to a text file and return the file path.

    Args:
        content (str): Text content to save.
        filename (str): Name of the output file.

    Returns:
        str: Absolute path to the saved file.
    """
    path = Path(filename)

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(content, encoding="utf-8")

    return str(path.resolve())