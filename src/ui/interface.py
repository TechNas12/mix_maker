import asyncio
import math
import os
from dotenv                             import load_dotenv
from src.ui.theme                       import console, COLORS, SYMBOLS
from src.ui.components.header           import print_header
from src.ui.components.progress         import PipelineProgress
from src.ui.components.logger           import EventLogger
from src.ui.components.output           import print_output, print_save_confirmation
from src.ui.components.inputs           import collect_inputs
from src.ui.components.music_panel      import MusicPanel
from src.provider.llm                   import LLM
from src.provider.music_generator       import MusicGenerator
from src.core.mix_maker                 import MixMaker
from src.utils.exception                import CustomException
from src.utils.logger                   import log
from src.utils.text                     import load_text_file
from src.utils.prompts_utils.style_retriever import get_style_by_id
from rich.rule                          import Rule
from pathlib                            import Path
from datetime                           import datetime
import time

load_dotenv()


# -------------------------------------------------------------------------
# LLM patch — wired to Gemini's _generate(system_prompt, user_content)
# -------------------------------------------------------------------------

def _patch_llm(llm: LLM, logger: EventLogger) -> LLM:
    original_generate = llm._generate  # Gemini: (system_prompt, user_content)

    def patched_generate(system_prompt: str, user_content: str):
        last_exception = None
        for attempt in range(1, llm.MAX_RETRIES + 1):
            try:
                logger.attempt(attempt, llm.MAX_RETRIES, llm.MODEL)
                response = original_generate(system_prompt, user_content)
                logger.model_success(llm.MODEL, attempt)
                log(f"Model '{llm.MODEL}' responded on attempt {attempt}.")
                return response
            except Exception as e:
                last_exception = e
                wait = llm.RETRY_DELAY * (2 ** (attempt - 1))
                logger.retry(
                    attempt = attempt,
                    total   = llm.MAX_RETRIES,
                    model   = llm.MODEL,
                    error   = str(e),
                    wait    = wait,
                )
                import time as t
                t.sleep(wait)

        raise CustomException(
            error   = last_exception,
            message = f"Model '{llm.MODEL}' failed after {llm.MAX_RETRIES} retries.",
            context = {"model": llm.MODEL, "max_retries": llm.MAX_RETRIES}
        )

    llm._generate = patched_generate
    return llm


# -------------------------------------------------------------------------
# Pipeline Runner
# -------------------------------------------------------------------------

def run_pipeline(
    user_prompt : str,
    style_id    : str,
    n           : int = 5,
) -> None:
    llm    = LLM(os.getenv("LLM_KEY"))
    logger = EventLogger()

    # ── Patch LLM ─────────────────────────────────────────────────────────
    llm = _patch_llm(llm, logger)

    # ── Load shared resources ─────────────────────────────────────────────
    try:
        system_prompt = load_text_file(llm.system_prompt_file_path)
        style         = get_style_by_id(style_id, llm.style_prompts_file_path)
    except CustomException as e:
        logger.pipeline_failed(reason=str(e.original_error))
        return

    # ── STAGE 1 — LLM batch loop ──────────────────────────────────────────
    total_batches  = math.ceil(n / LLM.BATCH_SIZE)
    all_variations = []
    batches_done   = 0

    logger.pipeline_start(n=n, total_batches=total_batches)

    with PipelineProgress(
        total_batches    = total_batches,
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
                    model     = llm.MODEL,
                )
                progress.advance(
                    batch_num  = batch_num + 1,
                    batch_size = len(batch_variations),
                    model      = llm.MODEL,
                )

            except CustomException as e:
                logger.batch_failed(
                    batch_num = batch_num + 1,
                    reason    = str(e.original_error),
                )
                break

        complete = len(all_variations) == n
        if complete:
            progress.complete()
        elif all_variations:
            progress.partial(collected=len(all_variations))
        else:
            progress.fail(reason="No variations collected")

    # Re-index
    for i, v in enumerate(all_variations, start=1):
        v["index"] = i

    llm_result = {
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

    if complete:
        logger.pipeline_complete(total=len(all_variations))
    else:
        logger.pipeline_partial(collected=len(all_variations), requested=n)

    print_output(llm_result)
    file_path = llm.save_response(llm_result)
    print_save_confirmation(str(file_path))

    if not all_variations:
        logger.pipeline_failed(reason="No variations to generate audio for.")
        return

    # ── STAGE 2 — Music Generation ────────────────────────────────────────
    console.print()
    console.print(
        Rule(
            title  = f"[primary] {SYMBOLS['info']} AUDIO GENERATION [/]",
            style  = COLORS["primary"],
            align  = "center",
        )
    )
    console.print()

    generator   = MusicGenerator(api_key=os.getenv("MUSIC_KEY"))
    music_panel = MusicPanel(variations=all_variations)
    music_result = None

    # ── Key fix: run async generation OUTSIDE the Live context ────────────
    # Rich's Live uses the event loop internally on Windows; asyncio.run()
    # inside a Live block causes CancelledError / KeyboardInterrupt.
    # Solution: collect results first via a plain asyncio loop, feed them
    # into the panel via on_complete, then open Live only for display.

    # Collected results buffer — on_complete stores here before Live starts
    _pending_results: list[dict] = []

    def _buffered_on_complete(result: dict) -> None:
        """Collect results before the Live panel is open."""
        _pending_results.append(result)

    run_id     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(generator.ARTIFACTS_DIR) / run_id

    try:
        # Run async generation fully before entering Live
        raw_results: list[dict] = asyncio.run(
            generator._generate_all_async(
                variations  = all_variations,
                output_dir  = output_dir,
                on_complete = _buffered_on_complete,
            )
        )
    except CustomException as e:
        log(f"Music generation failed: {e}", level="critical")
        console.print(
            f"\n  [error]{SYMBOLS['error']}  "
            f"Audio generation failed: {e.original_error}[/]\n"
        )
        return
    except Exception as e:
        log(f"Music generation failed unexpectedly: {e}", level="critical")
        console.print(
            f"\n  [error]{SYMBOLS['error']}  "
            f"Audio generation failed: {e}[/]\n"
        )
        return

    # Build music_result manifest (mirrors MusicGenerator.generate() output)
    succeeded    = sum(1 for r in raw_results if r["status"] == "success")
    failed       = len(raw_results) - succeeded
    music_result = {
        "run_id":      run_id,
        "style_id":    llm_result.get("style_id", ""),
        "base_prompt": llm_result.get("base_prompt", ""),
        "output_dir":  str(output_dir),
        "complete":    failed == 0,
        "results":     raw_results,
        "meta": {
            "total":     len(raw_results),
            "succeeded": succeeded,
            "failed":    failed,
        }
    }

    # Now open the Live panel and replay all buffered results instantly
    with music_panel:
        for result in _pending_results:
            music_panel.on_variation_complete(result)

    # ── STAGE 3 — Stitching ───────────────────────────────────────────────
    console.print()
    console.print(
        Rule(
            title  = f"[primary] {SYMBOLS['info']} STITCHING [/]",
            style  = COLORS["primary"],
            align  = "center",
        )
    )
    console.print()

    try:
        mix_maker    = MixMaker()
        final_result = mix_maker.stitch(
            music_result = music_result,
            crossfade_ms = 3000,
            gap_ms       = 800,
        )

        music_panel.add_stitch_result(final_result)

        from src.ui.components.output import print_stitch_confirmation
        print_stitch_confirmation(final_result)

        log(f"Stitching complete → {final_result['stitched']['file_path']}")

    except CustomException as e:
        log(f"Stitching failed: {e}", level="error")
        console.print(
            f"\n  [error]{SYMBOLS['error']}  "
            f"Stitching failed: {e.original_error}[/]\n"
        )


# -------------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------------

def run_pipeline_cli():
    from dotenv import load_dotenv
    import os

    _search_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent / ".env",
        Path.home() / ".mixmaker.env",
    ]

    print("\n[DEBUG] Searching for .env in:")
    for p in _search_paths:
        print(f"  {'✔' if p.exists() else '✘'}  {p}")

    loaded = False
    for env_path in _search_paths:
        if env_path.exists():
            load_dotenv(env_path)
            loaded = True
            print(f"[DEBUG] Loaded from: {env_path}")
            break

    if not loaded:
        load_dotenv()
        print("[DEBUG] Fallback load_dotenv() used")

    print(f"[DEBUG] LLM_KEY loaded:   {'YES' if os.getenv('LLM_KEY')   else 'NO'}")
    print(f"[DEBUG] MUSIC_KEY loaded: {'YES' if os.getenv('MUSIC_KEY') else 'NO'}\n")

    from src.ui.components.header import print_banner_only

    print_banner_only()

    # Use class-level path constant — no dummy instantiation needed
    user_prompt, style_id, n = collect_inputs(
        style_prompts_file=LLM.STYLE_PROMPTS_PATH
    )

    run_pipeline(
        user_prompt=user_prompt,
        style_id=style_id,
        n=n,
    )


if __name__ == "__main__":
    run_pipeline_cli()