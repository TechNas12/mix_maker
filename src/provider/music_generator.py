# src/music/music_generator.py

import asyncio
import os
from datetime   import datetime
from pathlib    import Path
from typing     import Callable

from dotenv         import load_dotenv
from google         import genai

from src.utils.exception    import CustomException
from src.utils.logger       import log
from src.utils.load_json    import save_json

load_dotenv()


class MusicGenerator:
    MODEL            = "lyria-3-pro-preview"
    ARTIFACTS_DIR    = "artifacts"
    CONCURRENT_LIMIT = 10

    MIME_EXT_MAP = {
        "audio/mpeg": "mp3",
        "audio/mp3":  "mp3",
        "audio/wav":  "wav",
        "audio/ogg":  "ogg",
        "audio/flac": "flac",
    }

    def __init__(self, api_key: str | None = None):
        """
        Initialize the Lyria client.

        Args:
            api_key (str | None): Google GenAI API key.
                                  Falls back to MUSIC_KEY env var if not provided.
        """
        resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise CustomException(
                error   = ValueError("No API key provided."),
                message = "MusicGenerator requires a Google GenAI API key.",
                context = {"source": "api_key param or MUSIC_KEY env var"}
            )
        self.client = genai.Client(api_key=resolved_key)

    # -------------------------------------------------------------------------
    # Internal — single variation
    # -------------------------------------------------------------------------

    async def _generate_single(
        self,
        variation   : dict,
        output_dir  : Path,
        semaphore   : asyncio.Semaphore,
        on_complete : Callable | None = None,
    ) -> dict:
        """
        Generate audio for one prompt variation and save it to disk.

        Response structure from Lyria:
            part[0] → text: track markers e.g. "[[A0]]\n[[B1]]"
            part[1] → inline_data: Blob(mime_type="audio/mpeg", data=<bytes>)

        Iterates all parts and takes the first one with valid inline_data
        so the code is resilient to part ordering changes.

        Args:
            variation   (dict):              Single variation dict.
            output_dir  (Path):              Directory to save audio file.
            semaphore   (asyncio.Semaphore): Limits concurrent API calls.
            on_complete (Callable|None):     UI callback fired on completion.

        Returns:
            dict: {
                "index":     int,
                "prompt":    str,
                "file":      str | None,
                "size_mb":   float | None,
                "mime_type": str | None,
                "status":    "success" | "failed",
                "error":     str | None,
            }
        """
        index  = variation.get("index")
        prompt = variation.get("prompt", "")

        log(f"Variation {index} — queued")

        async with semaphore:
            log(f"Variation {index} — generating (model: {self.MODEL})")

            try:
                response = await self.client.aio.models.generate_content(
                    model    = self.MODEL,
                    contents = prompt,
                )

                # ── Validate response structure ───────────────────────────
                if not response.candidates:
                    raise ValueError(
                        "No candidates returned. "
                        "Prompt may have been filtered by the safety system."
                    )

                candidate = response.candidates[0]

                if not candidate.content:
                    raise ValueError(
                        f"Candidate has no content. "
                        f"finish_reason: {candidate.finish_reason}"
                    )

                if not candidate.content.parts:
                    raise ValueError(
                        f"Content has no parts. "
                        f"finish_reason: {candidate.finish_reason}"
                    )

                # ── Find audio part ───────────────────────────────────────
                # part[0] = text track markers (not audio)
                # part[1] = actual audio blob
                # iterate all, take first with valid inline_data
                audio_bytes = None
                mime_type   = None

                for part in candidate.content.parts:
                    if (
                        hasattr(part, "inline_data")
                        and part.inline_data is not None
                        and part.inline_data.data is not None
                    ):
                        audio_bytes = part.inline_data.data
                        mime_type   = getattr(
                            part.inline_data, "mime_type", "audio/mpeg"
                        )
                        break

                if audio_bytes is None:
                    raise ValueError(
                        f"No inline audio data found in any part. "
                        f"Parts inspected: {len(candidate.content.parts)}"
                    )

                # ── Determine file extension from mime type ────────────────
                ext       = self.MIME_EXT_MAP.get(mime_type, "mp3")
                file_name = f"variation_{index}.{ext}"
                file_path = output_dir / file_name

                file_path.write_bytes(audio_bytes)
                size_mb = round(len(audio_bytes) / (1024 * 1024), 2)

                log(
                    f"Variation {index} saved → {file_path} "
                    f"({len(audio_bytes):,} bytes | {size_mb}MB | {mime_type})"
                )

                result = {
                    "index":     index,
                    "prompt":    prompt,
                    "file":      file_name,
                    "size_mb":   size_mb,
                    "mime_type": mime_type,
                    "status":    "success",
                    "error":     None,
                }

            except Exception as e:
                log(
                    f"Variation {index} failed: {type(e).__name__}: {e}",
                    level="error"
                )
                result = {
                    "index":     index,
                    "prompt":    prompt,
                    "file":      None,
                    "size_mb":   None,
                    "mime_type": None,
                    "status":    "failed",
                    "error":     f"{type(e).__name__}: {e}",
                }

        # ── Notify UI callback ────────────────────────────────────────────
        if on_complete is not None:
            try:
                on_complete(result)
            except Exception as cb_error:
                log(
                    f"on_complete callback error for variation {index}: {cb_error}",
                    level="warning"
                )

        return result

    # -------------------------------------------------------------------------
    # Internal — fan-out orchestrator
    # -------------------------------------------------------------------------

    async def _generate_all_async(
        self,
        variations  : list[dict],
        output_dir  : Path,
        on_complete : Callable | None = None,
    ) -> list[dict]:
        """
        Fan out all variations concurrently with a semaphore.

        Args:
            variations  (list[dict]):    All variation dicts from LLM result.
            output_dir  (Path):          Timestamped artifacts directory.
            on_complete (Callable|None): UI callback forwarded to each task.

        Returns:
            list[dict]: Results in input order.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        log(
            f"Output directory → {output_dir} | "
            f"Generating {len(variations)} variation(s) | "
            f"Concurrency: {self.CONCURRENT_LIMIT}"
        )

        semaphore  = asyncio.Semaphore(self.CONCURRENT_LIMIT)
        coroutines = [
            self._generate_single(variation, output_dir, semaphore, on_complete)
            for variation in variations
        ]

        results: list[dict] = await asyncio.gather(*coroutines)
        return list(results)

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def generate(
        self,
        llm_result  : dict,
        on_complete : Callable | None = None,
    ) -> dict:
        """
        Generate audio for all variations concurrently and save a manifest.

        Args:
            llm_result  (dict):          Output from LLM.generate_variations().
                                         Must contain a non-empty 'variations' list.
            on_complete (Callable|None): Optional UI callback per variation.
                                         Receives one result dict.
                                         Defaults to None — safe to omit.

        Returns:
            dict: {
                "run_id":      str,
                "style_id":    str,
                "base_prompt": str,
                "output_dir":  str,
                "complete":    bool,
                "results":     list[dict],
                "meta": {
                    "total":     int,
                    "succeeded": int,
                    "failed":    int,
                }
            }

        Raises:
            CustomException: If variations missing/empty or async pipeline fails.
        """
        variations = llm_result.get("variations")
        if not variations:
            raise CustomException(
                error   = ValueError("'variations' key is missing or empty."),
                message = "Invalid LLM result passed to MusicGenerator.generate().",
                context = {"llm_result_keys": list(llm_result.keys())}
            )

        run_id     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(self.ARTIFACTS_DIR) / run_id

        log(
            f"Music generation started — "
            f"run_id: {run_id} | "
            f"variations: {len(variations)} | "
            f"concurrency: {self.CONCURRENT_LIMIT}"
        )

        try:
            results = asyncio.run(
                self._generate_all_async(variations, output_dir, on_complete)
            )
        except Exception as e:
            raise CustomException(
                error   = e,
                message = "Async music generation pipeline failed unexpectedly.",
                context = {
                    "run_id":     run_id,
                    "output_dir": str(output_dir),
                }
            )

        succeeded = sum(1 for r in results if r["status"] == "success")
        failed    = len(results) - succeeded
        complete  = failed == 0

        if complete:
            log(f"All {succeeded}/{len(variations)} variations generated successfully.")
        else:
            log(
                f"Finished with {failed} failure(s) — "
                f"{succeeded}/{len(variations)} succeeded.",
                level="warning"
            )

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

        save_json(
            data      = final_result,
            file_name = "manifest",
            directory = str(output_dir),
            overwrite = True,
        )

        log(f"Manifest saved → {output_dir}/manifest.json")
        return final_result

    # -------------------------------------------------------------------------
    # Display — standalone use only, outside UI layer
    # -------------------------------------------------------------------------

    def display_manifest(self, manifest: dict) -> None:
        """
        Pretty-print a music generation manifest.
        For standalone use — UI layer has its own themed display.

        Args:
            manifest (dict): Result from MusicGenerator.generate().
        """
        from rich.console   import Console
        from rich.table     import Table
        from rich.panel     import Panel
        from rich.text      import Text
        from rich.columns   import Columns
        from rich           import box

        _console = Console()

        run_id      = manifest.get("run_id", "—")
        style_id    = manifest.get("style_id", "—")
        base_prompt = manifest.get("base_prompt", "—")
        output_dir  = manifest.get("output_dir", "—")
        complete    = manifest.get("complete", False)
        meta        = manifest.get("meta", {})
        results     = manifest.get("results", [])

        _console.print(Panel(
            Text.from_markup("\n".join([
                f"[dim]Run ID   :[/dim]  [bold cyan]{run_id}[/bold cyan]",
                f"[dim]Style    :[/dim]  [bold yellow]{style_id}[/bold yellow]",
                f"[dim]Output   :[/dim]  [dim italic]{output_dir}[/dim italic]",
                f"[dim]Status   :[/dim]  "
                + ("[bold green]✓ COMPLETE[/bold green]" if complete else "[bold red]✗ PARTIAL[/bold red]"),
            ])),
            title        = "[bold white] 🎵 Mix Maker — Music Generation Run [/bold white]",
            border_style = "cyan",
            padding      = (1, 2),
        ))

        _console.print(Panel(
            f"[italic white]{base_prompt}[/italic white]",
            title        = "[bold white]Base Prompt[/bold white]",
            border_style = "dim",
            padding      = (0, 2),
        ))

        table = Table(
            box          = box.ROUNDED,
            border_style = "dim",
            header_style = "bold cyan",
            show_lines   = True,
            expand       = True,
        )
        table.add_column("#",       style="bold white", width=4,  justify="center")
        table.add_column("Status",  width=12,           justify="center")
        table.add_column("File",    style="dim cyan",   width=24)
        table.add_column("Size",    style="dim",        width=8)
        table.add_column("Mime",    style="dim",        width=12)
        table.add_column("Prompt",  style="white",      ratio=1)
        table.add_column("Error",   style="red",        ratio=1)

        for r in sorted(results, key=lambda x: x.get("index", 0)):
            size_str = f"{r.get('size_mb')}MB" if r.get("size_mb") else "—"
            prompt   = r.get("prompt", "")
            error    = r.get("error") or "—"

            table.add_row(
                str(r.get("index", "?")),
                "[bold green]✓ success[/bold green]"
                if r.get("status") == "success"
                else "[bold red]✗ failed[/bold red]",
                r.get("file") or "—",
                size_str,
                r.get("mime_type") or "—",
                prompt[:100] + "…" if len(prompt) > 100 else prompt,
                error[:60]   + "…" if len(error)  > 60  else error,
            )

        _console.print(table)
        _console.print(Columns([
            Panel(
                f"[bold white]{meta.get('total', 0)}[/bold white]",
                title        = "[dim]Total[/dim]",
                border_style = "dim",
                padding      = (0, 3),
            ),
            Panel(
                f"[bold green]{meta.get('succeeded', 0)}[/bold green]",
                title        = "[dim]Succeeded[/dim]",
                border_style = "green",
                padding      = (0, 3),
            ),
            Panel(
                f"[bold red]{meta.get('failed', 0)}[/bold red]",
                title        = "[dim]Failed[/dim]",
                border_style = "red",
                padding      = (0, 3),
            ),
        ], equal=True, expand=True))
        _console.print()

    def generate_and_display(self, llm_result: dict) -> dict:
        """
        Convenience — generate and display. Standalone use only.

        Args:
            llm_result (dict): Output from LLM.generate_variations().

        Returns:
            dict: Full manifest result.
        """
        result = self.generate(llm_result)
        self.display_manifest(result)
        return result