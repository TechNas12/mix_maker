from src.utils.load_json import load_json


def get_style_by_id(style_id: str, file_path: str):
    """
    Get style description and directives by style ID.

    Args:
        style_id: Style ID (e.g. "cinematic", "balanced")
        file_path: Path to style_prompts.json

    Returns:
        dict containing id, label, description, directives

    Raises:
        ValueError if style is not found
    """

    styles = load_json(file_path)

    for style in styles:
        if style["id"] == style_id:
            return {
                "id": style["id"],
                "label": style["label"],
                "description": style["description"],
                "directives": style["directives"]
            }

    raise ValueError(f"Style '{style_id}' not found")