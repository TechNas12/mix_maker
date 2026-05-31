from src.utils.exception import CustomException
from src.utils.logger import log
from dotenv import load_dotenv
import os
import math
import time
import json
from openrouter import OpenRouter
from src.utils.prompts_utils.style_retriever import get_style_by_id
from src.utils.load_json import load_json, save_json, display_json
from src.utils.text import load_text_file

load_dotenv()
llm_key = os.getenv("LLM_KEY")


class LLM:
    PRIMARY_MODEL   = "nvidia/nemotron-3-super-120b-a12b:free"
    FALLBACK_MODEL  = "google/gemma-4-31b-it:free"
    MAX_RETRIES     = 3
    RETRY_DELAY     = 2   # seconds — doubles each attempt (exponential backoff)
    BATCH_SIZE      = 2   # max variations requested per API call

    def __init__(self, api_key: str):
        """
        Initialize the LLM provider and configure prompt resource paths.

        Args:
            api_key (str): OpenRouter API key.
        """
        self.api_key = api_key
        self.client  = OpenRouter(api_key)
        self.system_prompt_file_path: str = "src/prompts/system_prompt.txt"
        self.style_prompts_file_path: str = "src/prompts/style_prompts.json"

    # -------------------------------------------------------------------------
    # Message Building
    # -------------------------------------------------------------------------

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        style: dict,
        batch_size: int,
        start_index: int,
    ) -> list:
        """
        Construct the messages payload for a single batch API call.

        Args:
            system_prompt (str): Loaded system prompt text.
            user_prompt   (str): The user's input prompt.
            style         (dict): Style metadata — label, description, directives.
            batch_size    (int): How many variations to generate in this batch.
            start_index   (int): The variation index this batch should start from,
                                 so the model numbers them correctly across batches.

        Returns:
            list: Formatted messages list ready for the API.
        """
        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    f"Production Style: {style['label']}\n"
                    f"Production Description: {style['description']}\n"
                    f"Production Directives: {style['directives']}\n"
                    f"Number of Variations to Generate: {batch_size}\n"
                    f"Start Index: {start_index}\n\n"
                    f"User Prompt:\n{user_prompt}"
                )
            }
        ]

    # -------------------------------------------------------------------------
    # Core Generation — single attempt with retries + fallback
    # -------------------------------------------------------------------------

    def _try_generate(self, model: str, messages: list):
        """
        Attempt generation from a model with retry logic and exponential backoff.

        Args:
            model    (str):  Model identifier string.
            messages (list): Formatted messages payload.

        Returns:
            Raw API response object on success.

        Raises:
            CustomException: Wraps the last exception after all retries exhausted.
        """
        last_exception = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                log(f"Attempt {attempt}/{self.MAX_RETRIES} — model: {model}")
                response = self.client.chat.send(
                    model=model,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=4096,
                )
                log(f"Model '{model}' responded successfully on attempt {attempt}.")
                return response

            except Exception as e:
                last_exception = e
                wait = self.RETRY_DELAY * (2 ** (attempt - 1))  # 2s → 4s → 8s
                log(
                    f"Attempt {attempt} failed for model '{model}': {e}. "
                    f"Retrying in {wait}s...",
                    level="warning"
                )
                time.sleep(wait)

        raise CustomException(
            error=last_exception,
            message=f"Model '{model}' failed after {self.MAX_RETRIES} retries.",
            context={"model": model, "max_retries": self.MAX_RETRIES}
        )

    def _generate_with_fallback(self, messages: list):
        """
        Try the primary model first. On full failure, switch to the fallback model.

        Args:
            messages (list): Formatted messages payload.

        Returns:
            Raw API response from whichever model succeeded.

        Raises:
            CustomException: If both primary and fallback models fail.
        """
        primary_error = None

        # --- Primary ---
        try:
            return self._try_generate(self.PRIMARY_MODEL, messages)
        except CustomException as e:
            primary_error = e
            log(
                f"Primary model '{self.PRIMARY_MODEL}' exhausted all retries. "
                f"Switching to fallback '{self.FALLBACK_MODEL}'.\n{e}",
                level="error"
            )

        # --- Fallback ---
        try:
            return self._try_generate(self.FALLBACK_MODEL, messages)
        except CustomException as fallback_error:
            log(
                f"Fallback model '{self.FALLBACK_MODEL}' also failed.\n{fallback_error}",
                level="critical"
            )
            raise CustomException(
                error=fallback_error,
                message="Both primary and fallback models failed.",
                context={
                    "primary_model":  self.PRIMARY_MODEL,
                    "fallback_model": self.FALLBACK_MODEL,
                    "primary_error":  str(primary_error),
                    "fallback_error": str(fallback_error),
                }
            )

    # -------------------------------------------------------------------------
    # Batch Generation — single batch request + parse
    # -------------------------------------------------------------------------

    def _generate_batch(
        self,
        system_prompt: str,
        user_prompt: str,
        style: dict,
        batch_size: int,
        start_index: int,
    ) -> list[dict]:
        """
        Generate one batch of variations and return them as a clean list.

        Builds the prompt, calls the API with fallback support, parses the
        raw response, and returns only the variations list for this batch.

        Args:
            system_prompt (str): Loaded system prompt text.
            user_prompt   (str): The user's input prompt.
            style         (dict): Style metadata.
            batch_size    (int): Number of variations to generate in this batch.
            start_index   (int): Index the model should start numbering from.

        Returns:
            list[dict]: Parsed variations for this batch,
                        e.g. [{"index": 3, "prompt": "..."}, ...]

        Raises:
            CustomException: If generation or parsing fails for this batch.
        """
        messages = self._build_messages(
            system_prompt, user_prompt, style, batch_size, start_index
        )
        response  = self._generate_with_fallback(messages)
        parsed    = self._parse_raw_response(response)
        variations = parsed.get("variations", [])

        log(
            f"Batch starting at index {start_index} → "
            f"received {len(variations)} variation(s)."
        )
        return variations

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------

    def generate_response(self, user_prompt: str, style_id: str, n: int = 5) -> dict:
        """
        Generate n prompt variations using batched API calls (BATCH_SIZE per call).

        Iterates in batches of BATCH_SIZE until n variations are collected,
        stitches all batches into a single result, and re-indexes cleanly 1 → n.

        If a batch fails, collected variations so far are saved as a partial
        result rather than losing all progress.

        Args:
            user_prompt (str): The user's music description prompt.
            style_id    (str): Style identifier for loading directives.
            n           (int): Total number of variations to generate. Default: 5.

        Returns:
            dict: {
                "style_id":    str,
                "base_prompt": str,
                "complete":    bool,   # False if a batch failed mid-pipeline
                "variations":  list[dict],
                "meta": {
                    "total_variations_requested": int,
                    "total_variations_returned":  int,
                    "total_batches":              int,
                    "batches_completed":          int,
                }
            }

        Raises:
            CustomException: If loading prompts/style fails, or first batch fails.
        """
        if not isinstance(n, int) or n < 1:
            raise CustomException(
                error=ValueError(f"Invalid value for n: {n!r}"),
                message="'n' must be a positive integer.",
                context={"n": n}
            )

        # --- Load shared resources once ---
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

        total_batches     = math.ceil(n / self.BATCH_SIZE)
        all_variations    = []
        batches_completed = 0

        log(
            f"Starting pipeline — n={n}, batch_size={self.BATCH_SIZE}, "
            f"total_batches={total_batches}"
        )

        # --- Batch loop ---
        for batch_num in range(total_batches):
            start_index       = batch_num * self.BATCH_SIZE + 1
            remaining         = n - len(all_variations)
            this_batch_size   = min(self.BATCH_SIZE, remaining)

            log(
                f"Batch {batch_num + 1}/{total_batches} — "
                f"requesting {this_batch_size} variation(s) "
                f"starting at index {start_index}"
            )

            try:
                batch_variations = self._generate_batch(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    style=style,
                    batch_size=this_batch_size,
                    start_index=start_index,
                )
                all_variations.extend(batch_variations)
                batches_completed += 1

            except CustomException as e:
                log(
                    f"Batch {batch_num + 1} failed — stopping pipeline. "
                    f"Collected {len(all_variations)}/{n} variations so far.\n{e}",
                    level="error"
                )
                # Partial save — don't lose what we have
                break

        # --- Re-index cleanly 1 → n regardless of what the model returned ---
        for i, variation in enumerate(all_variations, start=1):
            variation["index"] = i

        complete = len(all_variations) == n

        if not complete:
            log(
                f"Pipeline completed partially — "
                f"{len(all_variations)}/{n} variations collected.",
                level="warning"
            )
        else:
            log(f"Pipeline complete — {len(all_variations)}/{n} variations collected.")

        return {
            "style_id":    style_id,
            "base_prompt": user_prompt,
            "complete":    complete,
            "variations":  all_variations,
            "meta": {
                "total_variations_requested": n,
                "total_variations_returned":  len(all_variations),
                "total_batches":              total_batches,
                "batches_completed":          batches_completed,
            }
        }

    # -------------------------------------------------------------------------
    # Response Parsing (internal — used by _generate_batch)
    # -------------------------------------------------------------------------

    def _parse_raw_response(self, response) -> dict:
        """
        Parse a raw API response object into a Python dict.

        Strips markdown code fences if present, decodes the JSON content,
        deduplicates variation entries, and returns the structured dict.

        Args:
            response: Raw response object from self.client.chat.send()

        Returns:
            dict: Parsed content with a 'variations' list.

        Raises:
            CustomException: If choices are empty, content is blank,
                             or content is not valid JSON.
        """
        try:
            if not response.choices:
                raise ValueError("Response returned an empty choices list.")

            choice      = response.choices[0]
            raw_content = choice.message.content

            if not raw_content or not raw_content.strip():
                raise ValueError("Response message content is empty.")

            # Strip markdown code fences
            cleaned = raw_content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                cleaned = cleaned[cleaned.index("\n") + 1:]
                cleaned = cleaned.rsplit("```", 1)[0]

            try:
                parsed_content = json.loads(cleaned)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Content is not valid JSON: {e}\nRaw: {raw_content}"
                )

            # Deduplicate — model occasionally emits duplicate index keys
            variations  = parsed_content.get("variations", [])
            seen        = set()
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
                message="Failed to parse LLM response.",
                context={
                    "model":         getattr(response, "model", "unknown"),
                    "finish_reason": (
                        response.choices[0].finish_reason
                        if response.choices else "no_choices"
                    ),
                }
            )

    def generate_variations(self, user_prompt: str, style_id: str, n: int = 5) -> dict:
        """
        Public entry point. Generates n prompt variations, displays and saves the result.

        Returns:
            dict: Stitched result with variations, meta, and complete flag.
        """
        result = self.generate_response(user_prompt, style_id, n)
        self.display_response(result, show_meta=False)
        self.save_response(result)
        return result
    
    # -------------------------------------------------------------------------
    # Save & Display (public — used after generate_response)
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

        Auto-generates filename from style_id + base_prompt if not provided.

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
            parsed     (dict): Result from generate_response().
            show_meta  (bool): Include the meta block. Default: True
        """
        display_data = parsed if show_meta else {
            k: v for k, v in parsed.items() if k != "meta"
        }
        style_id    = parsed.get("style_id", "response")
        base_prompt = parsed.get("base_prompt", "")
        title       = f"{style_id} — {base_prompt}" if base_prompt else style_id

        display_json(data=display_data, title=title)

