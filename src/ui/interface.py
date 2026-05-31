# src/ui/interface.py

import math
from src.ui.theme                   import console, COLORS, SYMBOLS
from src.ui.components.header       import print_header
from src.ui.components.progress     import PipelineProgress
from src.ui.components.logger       import EventLogger
from src.ui.components.output       import print_output, print_save_confirmation
from src.provider.llm               import LLM
from src.utils.exception            import CustomException
from src.utils.logger               import log
from dotenv                         import load_dotenv
import os
from src.ui.components.inputs import collect_inputs

load_dotenv()

# -------------------------------------------------------------------------
# Monkey-patch LLM internals to emit UI events
# -------------------------------------------------------------------------
# The LLM class has no knowledge of the UI layer.
# We wrap its private methods here so the UI gets notified
# of internal events (retries, fallbacks, model switches)
# without coupling LLM to Rich.
# -------------------------------------------------------------------------

def _patch_llm(llm: LLM, logger: EventLogger) -> LLM:
    """
    Wrap LLM._try_generate and LLM._generate_with_fallback with UI hooks.

    This keeps the LLM class clean while giving the UI full visibility
    into internal model-level events.

    Args:
        llm    (LLM):         The LLM instance to patch.
        logger (EventLogger): The active EventLogger instance.

    Returns:
        LLM: The same instance with patched methods.
    """
    original_try_generate        = llm._try_generate
    original_generate_fallback   = llm._generate_with_fallback

    def patched_try_generate(model: str, messages: list):
        last_exception = None

        import time
        for attempt in range(1, llm.MAX_RETRIES + 1):
            try:
                logger.attempt(attempt, llm.MAX_RETRIES, model)
                response = llm.client.chat.send(
                    model       = model,
                    messages    = messages,
                    temperature = 0.8,
                    max_tokens  = 4096,
                )
                logger.model_success(model, attempt)
                log(f"Model '{model}' responded successfully on attempt {attempt}.")
                return response

            except Exception as e:
                last_exception = e
                wait = llm.RETRY_DELAY * (2 ** (attempt - 1))
                logger.retry(
                    attempt = attempt,
                    total   = llm.MAX_RETRIES,
                    model   = model,
                    error   = str(e),
                    wait    = wait,
                )
                log(
                    f"Attempt {attempt} failed for model '{model}': {e}. "
                    f"Retrying in {wait}s...",
                    level="warning"
                )
                time.sleep(wait)

        raise CustomException(
            error   = last_exception,
            message = f"Model '{model}' failed after {llm.MAX_RETRIES} retries.",
            context = {"model": model, "max_retries": llm.MAX_RETRIES}
        )

    def patched_generate_fallback(messages: list):
        primary_error = None
        try:
            return patched_try_generate(llm.PRIMARY_MODEL, messages)
        except CustomException as e:
            primary_error = e
            logger.model_fallback(llm.PRIMARY_MODEL, llm.FALLBACK_MODEL)
            log(
                f"Primary model exhausted retries. Switching to fallback.",
                level="error"
            )

        try:
            return patched_try_generate(llm.FALLBACK_MODEL, messages)
        except CustomException as fallback_error:
            log(
                f"Fallback model also failed.",
                level="critical"
            )
            raise CustomException(
                error   = fallback_error,
                message = "Both primary and fallback models failed.",
                context = {
                    "primary_model":  llm.PRIMARY_MODEL,
                    "fallback_model": llm.FALLBACK_MODEL,
                    "primary_error":  str(primary_error),
                    "fallback_error": str(fallback_error),
                }
            )

    llm._try_generate             = patched_try_generate
    llm._generate_with_fallback   = patched_generate_fallback
    return llm


# -------------------------------------------------------------------------
# Pipeline Runner
# -------------------------------------------------------------------------

def run_pipeline(
    user_prompt : str,
    style_id    : str,
    n           : int = 5,
) -> None:
    """
    Run the full Mix Maker pipeline with a live terminal UI.

    Flow:
        1. Print header
        2. Patch LLM with UI hooks
        3. Load shared resources (system prompt + style)
        4. Run batch loop with live progress + event logs
        5. Stitch and re-index all variations
        6. Print output panel
        7. Save to disk + print save confirmation

    Args:
        user_prompt (str): Music description prompt from the user.
        style_id    (str): Style identifier e.g. 'cinematic'.
        n           (int): Total variations to generate. Default: 5.
    """
    llm    = LLM(os.getenv("LLM_KEY"))
    logger = EventLogger()

    # ── Header ────────────────────────────────────────────────────────────
    print_header(
        primary_model  = LLM.PRIMARY_MODEL,
        fallback_model = LLM.FALLBACK_MODEL,
        style_id       = style_id,
        user_prompt    = user_prompt,
        n              = n,
    )

    # ── Patch LLM with UI hooks ───────────────────────────────────────────
    llm = _patch_llm(llm, logger)

    # ── Load shared resources ─────────────────────────────────────────────
    from src.utils.text                         import load_text_file
    from src.utils.prompts_utils.style_retriever import get_style_by_id

    try:
        system_prompt = load_text_file(llm.system_prompt_file_path)
        style         = get_style_by_id(style_id, llm.style_prompts_file_path)
    except CustomException as e:
        logger.pipeline_failed(reason=str(e.original_error))
        console.print_exception()
        return

    # ── Batch loop ────────────────────────────────────────────────────────
    total_batches  = math.ceil(n / LLM.BATCH_SIZE)
    all_variations = []
    batches_done   = 0

    logger.pipeline_start(n=n, total_batches=total_batches)

    with PipelineProgress(
        total_batches   = total_batches,
        total_variations = n,
    ) as progress:

        for batch_num in range(total_batches):
            start_index     = batch_num * LLM.BATCH_SIZE + 1
            remaining       = n - len(all_variations)
            this_batch_size = min(LLM.BATCH_SIZE, remaining)

            logger.batch_start(
                batch_num   = batch_num + 1,
                total       = total_batches,
                size        = this_batch_size,
                start_index = start_index,
            )

            progress.set_description(
                f"Batch {batch_num + 1}/{total_batches} — "
                f"requesting {this_batch_size} variation(s)..."
            )

            try:
                batch_variations = llm._generate_batch(
                    system_prompt = system_prompt,
                    user_prompt   = user_prompt,
                    style         = style,
                    batch_size    = this_batch_size,
                    start_index   = start_index,
                )

                all_variations.extend(batch_variations)
                batches_done += 1

                logger.batch_success(
                    batch_num = batch_num + 1,
                    received  = len(batch_variations),
                    model     = llm.PRIMARY_MODEL,
                )

                progress.advance(
                    batch_num  = batch_num + 1,
                    batch_size = len(batch_variations),
                    model      = llm.PRIMARY_MODEL,
                )

            except CustomException as e:
                logger.batch_failed(
                    batch_num = batch_num + 1,
                    reason    = str(e.original_error),
                )
                log(
                    f"Batch {batch_num + 1} failed. "
                    f"Collected {len(all_variations)}/{n} so far.",
                    level="error"
                )
                break

        # ── Mark progress final state ─────────────────────────────────────
        complete = len(all_variations) == n
        if complete:
            progress.complete()
        elif all_variations:
            progress.partial(collected=len(all_variations))
        else:
            progress.fail(reason="No variations collected")

    # ── Re-index cleanly 1 → n ────────────────────────────────────────────
    for i, variation in enumerate(all_variations, start=1):
        variation["index"] = i

    # ── Assemble final result ─────────────────────────────────────────────
    result = {
        "style_id":    style_id,
        "base_prompt": user_prompt,
        "complete":    complete,
        "variations":  all_variations,
        "meta": {
            "total_variations_requested": n,
            "total_variations_returned":  len(all_variations),
            "total_batches":              total_batches,
            "batches_completed":          batches_done,
        }
    }

    # ── Log pipeline outcome ──────────────────────────────────────────────
    if complete:
        logger.pipeline_complete(total=len(all_variations))
    else:
        logger.pipeline_partial(
            collected = len(all_variations),
            requested = n,
        )

    # ── Output panel ──────────────────────────────────────────────────────
    print_output(result)

    # ── Save ──────────────────────────────────────────────────────────────
    if all_variations:
        file_path = llm.save_response(result)
        print_save_confirmation(str(file_path))


# -------------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------------

# replace the bottom of interface.py

if __name__ == "__main__":
    from src.provider.llm import LLM

    user_prompt, style_id, n = collect_inputs(
        style_prompts_file = LLM(os.getenv("LLM_KEY")).style_prompts_file_path
    )

    run_pipeline(
        user_prompt = user_prompt,
        style_id    = style_id,
        n           = n,
    )