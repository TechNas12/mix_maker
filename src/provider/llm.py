from src.utils.exception import CustomException
from src.utils.logger import log
from src.utils.prompts_utils.style_retriever import get_style_by_id
from src.utils.load_json import save_json, display_json
from src.utils.text import load_text_file
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time
import json

load_dotenv()
gemini_key = os.getenv("GOOGLE_API_KEY")


class LLM:
    MODEL       = "gemini-3.1-flash-lite"  # GA — cheapest, best for high-volume creative tasks
    MAX_RETRIES = 3
    RETRY_DELAY = 2   # seconds — doubles each attempt (exponential backoff)
    BATCH_SIZE  = 5   # variations requested per API call

    # Class-level path constants — avoids dummy instantiation just to read paths
    SYSTEM_PROMPT_PATH = "src/prompts/system_prompt.txt"
    STYLE_PROMPTS_PATH = "src/prompts/style_prompts.json"

    def __init__(self, api_key: str):
        """
        Initialize the Google Gemini client and configure prompt resource paths.

        Args:
            api_key (str): Google AI Studio API key.
        """
        self.api_key = api_key
        self.client  = genai.Client(api_key=api_key)

        # Instance aliases pointing at class-level constants (keeps backward compat)
        self.system_prompt_file_path: str = self.SYSTEM_PROMPT_PATH
        self.style_prompts_file_path: str = self.STYLE_PROMPTS_PATH

    # -------------------------------------------------------------------------
    # Message Building
    # -------------------------------------------------------------------------

    def _build_user_content(
        self,
        user_prompt: str,
        style: dict,
        n: int,
    ) -> str:
        """
        Construct the user content string for the API call.

        Args:
            user_prompt (str):  The user's input prompt.
            style       (dict): Style metadata — label, description, directives.
            n           (int):  Total number of variations to generate.

        Returns:
            str: Formatted user content string.
        """
        return (
            f"Production Style: {style['label']}\n"
            f"Production Description: {style['description']}\n"
            f"Production Directives: {style['directives']}\n"
            f"Number of Variations to Generate: {n}\n\n"
            f"User Prompt:\n{user_prompt}"
        )

    # -------------------------------------------------------------------------
    # Core Generation — single call with retries
    # -------------------------------------------------------------------------

    def _generate(self, system_prompt: str, user_content: str) -> any:
        """
        Call the Gemini API with retry logic and exponential backoff.

        Args:
            system_prompt (str): Loaded system prompt text.
            user_content  (str): Formatted user content string.

        Returns:
            Raw API response object on success.

        Raises:
            CustomException: Wraps the last exception after all retries exhausted.
        """
        last_exception = None

        config = types.GenerateContentConfig(
            temperature=0.8,
            max_output_tokens=4096,
            system_instruction=system_prompt,
        )

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                log(f"Attempt {attempt}/{self.MAX_RETRIES} — model: {self.MODEL}")

                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=user_content,
                    config=config,
                )

                log(f"Model responded successfully on attempt {attempt}.")
                return response

            except Exception as e:
                last_exception = e
                wait = self.RETRY_DELAY * (2 ** (attempt - 1))  # 2s → 4s → 8s
                log(
                    f"Attempt {attempt} failed: {e}. Retrying in {wait}s...",
                    level="warning"
                )
                time.sleep(wait)

        raise CustomException(
            error=last_exception,
            message=f"Model '{self.MODEL}' failed after {self.MAX_RETRIES} retries.",
            context={"model": self.MODEL, "max_retries": self.MAX_RETRIES}
        )

    # -------------------------------------------------------------------------
    # Batch Generation — used by the pipeline's batch loop
    # -------------------------------------------------------------------------

    def _generate_batch(
        self,
        system_prompt: str,
        user_prompt:   str,
        style:         dict,
        batch_size:    int,
        start_index:   int,
    ) -> list:
        """
        Generate a single batch of variations and return the variation list.

        Args:
            system_prompt (str):  Loaded system prompt text.
            user_prompt   (str):  The user's music description prompt.
            style         (dict): Style metadata dict.
            batch_size    (int):  How many variations to request in this batch.
            start_index   (int):  1-based index offset for re-indexing.

        Returns:
            list[dict]: Parsed variation dicts for this batch.

        Raises:
            CustomException: On generation or parse failure.
        """
        user_content = self._build_user_content(user_prompt, style, batch_size)
        response     = self._generate(system_prompt, user_content)
        parsed       = self._parse_raw_response(response)
        variations   = parsed.get("variations", [])

        # Re-index relative to start_index
        for i, v in enumerate(variations, start=start_index):
            v["index"] = i

        return variations

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------

    def generate_response(self, user_prompt: str, style_id: str, n: int = 5) -> dict:
        """
        Generate n prompt variations in a single API call.

        Args:
            user_prompt (str): The user's music description prompt.
            style_id    (str): Style identifier for loading directives.
            n           (int): Total number of variations to generate. Default: 5.

        Returns:
            dict: {
                "style_id":    str,
                "base_prompt": str,
                "variations":  list[dict],
                "meta": {
                    "model":                      str,
                    "total_variations_requested": int,
                    "total_variations_returned":  int,
                }
            }

        Raises:
            CustomException: If loading prompts/style fails, or generation fails.
        """
        if not isinstance(n, int) or n < 1:
            raise CustomException(
                error=ValueError(f"Invalid value for n: {n!r}"),
                message="'n' must be a positive integer.",
                context={"n": n}
            )

        # --- Load shared resources ---
        try:
            system_prompt = load_text_file(self.system_prompt_file_path)
            style         = get_style_by_id(style_id, self.style_prompts_file_path)
        except Exception as e:
            raise CustomException(
                error=e,
                message="Failed to load prompts or style configuration.",
                context={
                    "style_id":           style_id,
                    "system_prompt_file": self.system_prompt_file_path,
                    "style_prompts_file": self.style_prompts_file_path,
                }
            )

        log(f"Starting generation — model: {self.MODEL}, n={n}, style: {style_id}")

        user_content = self._build_user_content(user_prompt, style, n)
        response     = self._generate(system_prompt, user_content)
        parsed       = self._parse_raw_response(response)
        variations   = parsed.get("variations", [])

        # Re-index cleanly 1 → n regardless of what the model returned
        for i, variation in enumerate(variations, start=1):
            variation["index"] = i

        log(f"Generation complete — {len(variations)}/{n} variations returned.")

        return {
            "style_id":    style_id,
            "base_prompt": user_prompt,
            "variations":  variations,
            "meta": {
                "model":                      self.MODEL,
                "total_variations_requested": n,
                "total_variations_returned":  len(variations),
            }
        }

    # -------------------------------------------------------------------------
    # Response Parsing
    # -------------------------------------------------------------------------

    def _parse_raw_response(self, response) -> dict:
        """
        Parse a raw Gemini API response object into a Python dict.

        Strips markdown code fences if present, decodes the JSON content,
        deduplicates variation entries, and returns the structured dict.

        Args:
            response: Raw response object from self.client.models.generate_content()

        Returns:
            dict: Parsed content with a 'variations' list.

        Raises:
            CustomException: If content is blank or not valid JSON.
        """
        try:
            raw_content = response.text

            if not raw_content or not raw_content.strip():
                raise ValueError("Response text is empty.")

            # Strip markdown code fences
            cleaned = raw_content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                cleaned = cleaned[cleaned.index("\n") + 1:]
                cleaned = cleaned.rsplit("```", 1)[0]

            try:
                parsed_content = json.loads(cleaned)
            except json.JSONDecodeError as e:
                raise ValueError(f"Content is not valid JSON: {e}\nRaw: {raw_content}")

            # Deduplicate — model occasionally emits duplicate index keys
            variations = parsed_content.get("variations", [])
            seen       = set()
            clean_variations = []
            for v in variations:
                key = (v.get("index"), v.get("prompt", "")[:40])
                if key not in seen:
                    seen.add(key)
                    clean_variations.append(v)

            parsed_content["variations"] = clean_variations
            return parsed_content

        except Exception as e:
            raise CustomException(
                error=e,
                message="Failed to parse Gemini response.",
                context={
                    "model":         self.MODEL,
                    "finish_reason": str(
                        response.candidates[0].finish_reason
                        if response.candidates else "no_candidates"
                    ),
                }
            )

    # -------------------------------------------------------------------------
    # Public Entry Point
    # -------------------------------------------------------------------------

    def generate_variations(self, user_prompt: str, style_id: str, n: int = 5) -> dict:
        """
        Public entry point. Generates n prompt variations, displays and saves the result.

        Returns:
            dict: Result with variations and meta.
        """
        result = self.generate_response(user_prompt, style_id, n)
        self.display_response(result, show_meta=False)
        self.save_response(result)
        return result

    # -------------------------------------------------------------------------
    # Save & Display
    # -------------------------------------------------------------------------

    def save_response(
        self,
        parsed: dict,
        file_name: str | None = None,
        directory: str = "output",
        overwrite: bool = True,
    ) -> None:
        """
        Persist a generate_response() result to disk as a JSON file.

        Args:
            parsed    (dict): Result from generate_response().
            file_name (str):  Target filename. Auto-generated if omitted.
            directory (str):  Output folder. Created if missing. Default: 'output'
            overwrite (bool): Overwrite existing file. Default: True
        """
        if not file_name:
            style_id    = parsed.get("style_id", "response")
            base_prompt = parsed.get("base_prompt", "")
            slug        = base_prompt.lower().strip().replace(" ", "_")[:40]
            file_name   = f"{style_id}_{slug}" if slug else style_id

        save_json(data=parsed, file_name=file_name, directory=directory, overwrite=overwrite)

    def display_response(self, parsed: dict, show_meta: bool = True) -> None:
        """
        Pretty-print a generate_response() result using Rich.

        Args:
            parsed    (dict): Result from generate_response().
            show_meta (bool): Include the meta block. Default: True
        """
        display_data = parsed if show_meta else {
            k: v for k, v in parsed.items() if k != "meta"
        }
        style_id    = parsed.get("style_id", "response")
        base_prompt = parsed.get("base_prompt", "")
        title       = f"{style_id} — {base_prompt}" if base_prompt else style_id

        display_json(data=display_data, title=title)