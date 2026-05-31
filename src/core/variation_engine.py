

import json
import logging
from pathlib import Path
from typing import Optional

from provider.llm import LLMProvider, VariationRequest, VariationResponse
from config.llm_config import VARIATION_CONFIG

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class VariationEngine:
    """
    Knows where prompts and styles live on disk.
    Knows the validation rules.
    Delegates the actual LLM call to LLMProvider.
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or LLMProvider()
        self._system_prompt = self._load_system_prompt()
        self._styles = self._load_styles()

    def list_styles(self) -> list[dict]:
        return [
            {"id": s["id"], "label": s["label"], "description": s["description"]}
            for s in self._styles.values()
        ]

    def generate_variations(
        self,
        base_prompt: str,
        style_id: str,
        num_variations: int,
    ) -> VariationResponse:
        self._validate_inputs(base_prompt, style_id, num_variations)

        request = VariationRequest(
            base_prompt=base_prompt,
            style=self._styles[style_id],
            num_variations=num_variations,
            system_prompt=self._system_prompt,
        )

        logger.info(f"Requesting {num_variations} variation(s) | style: {style_id}")
        response = self.provider.get_variations(request)
        logger.info(f"Received {len(response.variations)} variation(s).")
        return response

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_system_prompt(self) -> str:
        path = PROMPTS_DIR / "system_prompt.txt"
        if not path.exists():
            raise FileNotFoundError(f"System prompt not found at {path}")
        return path.read_text(encoding="utf-8").strip()

    def _load_styles(self) -> dict[str, dict]:
        path = PROMPTS_DIR / "style_prompts.json"
        if not path.exists():
            raise FileNotFoundError(f"Style prompts not found at {path}")
        styles_list = json.loads(path.read_text(encoding="utf-8"))
        return {s["id"]: s for s in styles_list}

    def _validate_inputs(self, base_prompt: str, style_id: str, num_variations: int):
        if not base_prompt or not base_prompt.strip():
            raise ValueError("base_prompt cannot be empty.")

        if style_id not in self._styles:
            available = ", ".join(self._styles.keys())
            raise ValueError(f"Unknown style_id '{style_id}'. Available: {available}")

        min_v = VARIATION_CONFIG["min_variations"]
        max_v = VARIATION_CONFIG["max_variations"]
        if not (min_v <= num_variations <= max_v):
            raise ValueError(f"num_variations must be {min_v}–{max_v}, got {num_variations}.")