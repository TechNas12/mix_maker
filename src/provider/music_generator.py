import asyncio
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box

from src.utils.exception import CustomException
from src.utils.logger import log
from src.utils.load_json import save_json

console = Console()

load_dotenv()


class MusicGenerator:
    MODEL         = "lyria-3-pro-preview"
    AUDIO_EXT     = "mp3"
    ARTIFACTS_DIR = "artifacts"

    def __init__(self, api_key: str | None = None):
        """
        Initialize the Lyria client.

        Args:
            api_key (str | None): Google GenAI API key.
                                  Falls back to MUSIC_KEY env var if not provided.
        """
        resolved_key = api_key or os.getenv("MUSIC_KEY")
        self.client  = genai.Client(api_key=resolved_key)

    # -------------------------------------------------------------------------
    # Internal — single variation
    # -------------------------------------------------------------------------

    async def _generate_single(
        self,
        variation: dict,
        output_dir: Path,
    ) -> dict:
        """
        Generate audio for one prompt variation and save it to disk.

        Uses the SDK's native async client (client.aio.models.generate_content)
        so the event loop is never blocked. Each variation is fully independent
        — exceptions are caught and returned as a failed result rather than
        propagated, so one failure never cancels sibling tasks.

        Args:
            variation  (dict): Single variation dict with keys 'index' and 'prompt'.
            output_dir (Path): Directory where the audio file will be saved.

        Returns:
            dict: {
                "index":     int,
                "prompt":    str,
                "file":      str | None,   # filename on success, None on failure
                "status":    "success" | "failed",
                "error":     str | None,   # populated only on failure
            }
        """
        index     = variation.get("index")
        prompt    = variation.get("prompt", "")
        file_name = f"variation_{index}.{self.AUDIO_EXT}"
        file_path = output_dir / file_name

        log(f"Generating variation {index} — model: {self.MODEL}")

        try:
            # Native async SDK call — no thread wrapping needed
            response = await self.client.aio.models.generate_content(
                model=self.MODEL,
                contents=prompt,
            )

            # Extract raw audio bytes from the last part
            audio_bytes: bytes = response.parts[-1].inline_data.data

            # Write to disk
            file_path.write_bytes(audio_bytes)
            log(
                f"Variation {index} saved → {file_path} "
                f"({len(audio_bytes):,} bytes)"
            )

            return {
                "index":  index,
                "prompt": prompt,
                "file":   file_name,
                "status": "success",
                "error":  None,
            }

        except Exception as e:
            log(
                f"Variation {index} failed: {type(e).__name__}: {e}",
                level="error"
            )
            return {
                "index":  index,
                "prompt": prompt,
                "file":   None,
                "status": "failed",
                "error":  f"{type(e).__name__}: {e}",
            }

    # -------------------------------------------------------------------------
    # Internal — fan-out orchestrator
    # -------------------------------------------------------------------------

    async def _generate_all_async(
        self,
        variations: list[dict],
        output_dir: Path,
    ) -> list[dict]:
        """
        Fan out all variations concurrently and collect results.

        Creates the output directory, builds one coroutine per variation,
        then fires all of them simultaneously with asyncio.gather().
        Results come back in the same order as the input list.

        Args:
            variations (list[dict]): All variation dicts from the LLM result.
            output_dir (Path):       Timestamped artifacts directory to create.

        Returns:
            list[dict]: One result dict per variation, ordered by index.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"Output directory created → {output_dir} | "
            f"Generating {len(variations)} variation(s) concurrently."
        )

        coroutines = [
            self._generate_single(variation, output_dir)
            for variation in variations
        ]

        # gather() fires all coroutines at once and waits for ALL to finish
        # return_exceptions=False is fine here because _generate_single
        # never raises — it catches internally and returns a failed dict
        results: list[dict] = await asyncio.gather(*coroutines)
        return list(results)

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def generate(self, llm_result: dict) -> dict:
        """
        Public entry point. Takes the full LLM pipeline result, generates
        audio for every variation concurrently, saves a manifest, and returns
        a structured result.

        Calling code stays fully synchronous — asyncio.run() is managed here.

        Args:
            llm_result (dict): Output from LLM.generate_response() or
                               LLM.generate_variations(). Must contain
                               a 'variations' key with a non-empty list.

        Returns:
            dict: {
                "run_id":     str,          # timestamp string used as folder name
                "style_id":   str,
                "base_prompt": str,
                "output_dir": str,          # relative path to artifacts folder
                "complete":   bool,         # True only when failed == 0
                "results":    list[dict],   # one entry per variation
                "meta": {
                    "total":     int,
                    "succeeded": int,
                    "failed":    int,
                }
            }

        Raises:
            CustomException: If 'variations' key is missing or empty,
                             or if the async pipeline itself errors unexpectedly.
        """
        # --- Validate input ---
        variations = llm_result.get("variations")
        if not variations:
            raise CustomException(
                error=ValueError("'variations' key is missing or empty."),
                message="Invalid LLM result passed to MusicGenerator.generate().",
                context={"llm_result_keys": list(llm_result.keys())}
            )

        # --- Build timestamped output dir ---
        run_id     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(self.ARTIFACTS_DIR) / run_id

        log(
            f"Music generation started — run_id: {run_id} | "
            f"variations: {len(variations)}"
        )

        # --- Run async fan-out synchronously ---
        try:
            results = asyncio.run(
                self._generate_all_async(variations, output_dir)
            )
        except Exception as e:
            raise CustomException(
                error=e,
                message="Async music generation pipeline failed unexpectedly.",
                context={"run_id": run_id, "output_dir": str(output_dir)}
            )

        # --- Compute meta ---
        succeeded = sum(1 for r in results if r["status"] == "success")
        failed    = len(results) - succeeded
        complete  = failed == 0

        if complete:
            log(f"All {succeeded}/{len(variations)} variations generated successfully.")
        else:
            log(
                f"Pipeline finished with {failed} failure(s) — "
                f"{succeeded}/{len(variations)} succeeded.",
                level="warning"
            )

        # --- Assemble final result ---
        final_result = {
            "run_id":      run_id,
            "style_id":    llm_result.get("style_id", ""),
            "base_prompt": llm_result.get("base_prompt", ""),
            "output_dir":  str(output_dir),
            "complete":    complete,
            "results":     results,
            "meta": {
                "total":     len(variations),
                "succeeded": succeeded,
                "failed":    failed,
            }
        }

        # --- Save manifest into the same artifacts folder ---
        save_json(
            data=final_result,
            file_name="manifest",
            directory=str(output_dir),
            overwrite=True,
        )

        log(f"Manifest saved → {output_dir}/manifest.json")
        return final_result

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def display_manifest(self, manifest: dict) -> None:
        """
        Pretty-print a music generation manifest using Rich.

        Renders a header panel with run metadata, a per-variation results
        table with colour-coded status, and a summary footer.

        Args:
            manifest (dict): Result from MusicGenerator.generate().
        """
        run_id      = manifest.get("run_id", "—")
        style_id    = manifest.get("style_id", "—")
        base_prompt = manifest.get("base_prompt", "—")
        output_dir  = manifest.get("output_dir", "—")
        complete    = manifest.get("complete", False)
        meta        = manifest.get("meta", {})
        results     = manifest.get("results", [])

        # ── Header panel ──────────────────────────────────────────────────────
        complete_badge = (
            "[bold green]✓ COMPLETE[/bold green]"
            if complete
            else "[bold red]✗ PARTIAL[/bold red]"
        )

        header_lines = [
            f"[dim]Run ID   :[/dim]  [bold cyan]{run_id}[/bold cyan]",
            f"[dim]Style    :[/dim]  [bold yellow]{style_id}[/bold yellow]",
            f"[dim]Output   :[/dim]  [dim italic]{output_dir}[/dim italic]",
            f"[dim]Status   :[/dim]  {complete_badge}",
        ]

        header_text = Text.from_markup("\n".join(header_lines))
        console.print(
            Panel(
                header_text,
                title="[bold white] 🎵 Mix Maker — Music Generation Run [/bold white]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # ── Base prompt panel ─────────────────────────────────────────────────
        console.print(
            Panel(
                f"[italic white]{base_prompt}[/italic white]",
                title="[bold white]Base Prompt[/bold white]",
                border_style="dim",
                padding=(0, 2),
            )
        )

        # ── Variations table ──────────────────────────────────────────────────
        table = Table(
            box=box.ROUNDED,
            border_style="dim",
            header_style="bold cyan",
            show_lines=True,
            expand=True,
        )

        table.add_column("#",      style="bold white", width=4,  justify="center")
        table.add_column("Status", width=12,           justify="center")
        table.add_column("File",   style="dim cyan",   width=20)
        table.add_column("Prompt", style="white",      ratio=1)
        table.add_column("Error",  style="red",        ratio=1)

        for r in sorted(results, key=lambda x: x.get("index", 0)):
            idx    = str(r.get("index", "?"))
            status = r.get("status", "unknown")
            file   = r.get("file") or "—"
            prompt = r.get("prompt", "")
            error  = r.get("error") or "—"

            prompt_display = prompt[:120] + "…" if len(prompt) > 120 else prompt
            error_display  = error[:80]  + "…" if len(error) > 80  else error

            status_cell = (
                "[bold green]✓ success[/bold green]"
                if status == "success"
                else "[bold red]✗ failed[/bold red]"
            )

            table.add_row(idx, status_cell, file, prompt_display, error_display)

        console.print(table)

        # ── Summary footer ────────────────────────────────────────────────────
        total     = meta.get("total", 0)
        succeeded = meta.get("succeeded", 0)
        failed    = meta.get("failed", 0)

        summary_parts = [
            Panel(
                f"[bold white]{total}[/bold white]",
                title="[dim]Total[/dim]",
                border_style="dim",
                padding=(0, 3),
            ),
            Panel(
                f"[bold green]{succeeded}[/bold green]",
                title="[dim]Succeeded[/dim]",
                border_style="green",
                padding=(0, 3),
            ),
            Panel(
                f"[bold red]{failed}[/bold red]",
                title="[dim]Failed[/dim]",
                border_style="red",
                padding=(0, 3),
            ),
        ]

        console.print(Columns(summary_parts, equal=True, expand=True))
        console.print()

    def generate_and_display(self, llm_result: dict) -> dict:
        """
        Convenience method — generate audio and immediately display the manifest.

        Args:
            llm_result (dict): Output from LLM.generate_variations().

        Returns:
            dict: The full manifest result.
        """
        result = self.generate(llm_result)
        self.display_manifest(result)
        return result