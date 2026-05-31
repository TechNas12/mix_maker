from pathlib import Path

from pydub import AudioSegment
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.utils.exception import CustomException
from src.utils.logger import log
from src.utils.load_json import save_json, load_json

console = Console()


class MixMaker:
    OUTPUT_FILE  = "stitched_output.mp3"

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def stitch(
        self,
        music_result: dict,
        gap_ms: int = 0,
        crossfade_ms: int = 0,
    ) -> dict:
        """
        Stitch all successful variation audio files from a music generation
        run into a single WAV file.

        Reads source files directly from music_result['output_dir'].
        Saves the stitched file into the same directory.
        Updates and re-saves manifest.json with a 'stitched' block.

        Priority rule: if both gap_ms and crossfade_ms are provided,
        crossfade_ms takes priority and gap_ms is silently ignored.

        Args:
            music_result  (dict): Output from MusicGenerator.generate()
                                  or generate_and_display(). Must contain
                                  'output_dir' and 'results' keys.
            gap_ms        (int):  Milliseconds of silence to insert between
                                  tracks. Default: 0. Ignored if crossfade_ms > 0.
            crossfade_ms  (int):  Milliseconds of crossfade overlap between
                                  tracks. Default: 0. Takes priority over gap_ms.

        Returns:
            dict: Updated manifest with a 'stitched' block added:
                {
                    "file":            str,   # filename of stitched output
                    "file_path":       str,   # full relative path
                    "tracks_included": int,
                    "tracks_skipped":  int,
                    "gap_ms":          int,   # 0 if crossfade was used
                    "crossfade_ms":    int,   # 0 if gap was used
                    "mode":            str,   # "crossfade" | "gap" | "direct"
                }

        Raises:
            CustomException: If output_dir is missing, fewer than 2 valid
                             tracks exist, or audio processing fails.
        """
        # --- Validate input ---
        output_dir_str = music_result.get("output_dir")
        results        = music_result.get("results", [])

        if not output_dir_str:
            raise CustomException(
                error=ValueError("'output_dir' key missing from music_result."),
                message="Invalid music_result passed to MixMaker.stitch().",
                context={"music_result_keys": list(music_result.keys())}
            )

        output_dir = Path(output_dir_str)

        # --- Resolve mode — crossfade takes priority ---
        if crossfade_ms > 0:
            mode         = "crossfade"
            effective_gap = 0
            log(
                f"Mode: crossfade ({crossfade_ms}ms)"
                + (f" — gap_ms={gap_ms} ignored." if gap_ms > 0 else ".")
            )
        elif gap_ms > 0:
            mode              = "gap"
            effective_gap     = gap_ms
            crossfade_ms      = 0
            log(f"Mode: gap ({gap_ms}ms).")
        else:
            mode          = "direct"
            effective_gap = 0
            log("Mode: direct concatenation (no gap, no crossfade).")

        # --- Collect valid tracks in index order ---
        successful = [
            r for r in sorted(results, key=lambda x: x.get("index", 0))
            if r.get("status") == "success" and r.get("file")
        ]
        skipped = len(results) - len(successful)

        if skipped > 0:
            log(
                f"{skipped} variation(s) skipped (failed status or missing file).",
                level="warning"
            )

        if len(successful) < 2:
            raise CustomException(
                error=ValueError(
                    f"Need at least 2 successful tracks to stitch, "
                    f"got {len(successful)}."
                ),
                message="Not enough successful tracks to stitch.",
                context={
                    "successful_tracks": len(successful),
                    "total_tracks":      len(results),
                }
            )

        # --- Load and stitch ---
        try:
            log(f"Loading {len(successful)} track(s) from {output_dir} ...")

            segments = []
            for r in successful:
                file_path = output_dir / r["file"]
                log(f"  Loading → {file_path}")
                segment = AudioSegment.from_file(str(file_path))
                segments.append(segment)
                log(
                    f"  Loaded  → {r['file']} "
                    f"({len(segment) / 1000:.1f}s, "
                    f"{segment.frame_rate}Hz, "
                    f"{segment.channels}ch)"
                )

            log("Stitching tracks ...")
            stitched = self._stitch_segments(
                segments=segments,
                mode=mode,
                gap_ms=effective_gap,
                crossfade_ms=crossfade_ms,
            )

            # --- Export ---
            output_path = output_dir / self.OUTPUT_FILE
            stitched.export(str(output_path), format="mp3")

            duration_s = len(stitched) / 1000
            log(
                f"Stitched file saved → {output_path} "
                f"(duration: {duration_s:.1f}s)"
            )

        except CustomException:
            raise
        except Exception as e:
            raise CustomException(
                error=e,
                message="Audio stitching failed during processing.",
                context={
                    "output_dir":     str(output_dir),
                    "tracks":         len(successful),
                    "mode":           mode,
                }
            )

        # --- Build stitched block ---
        stitched_block = {
            "file":            self.OUTPUT_FILE,
            "file_path":       str(output_path),
            "tracks_included": len(successful),
            "tracks_skipped":  skipped,
            "gap_ms":          effective_gap,
            "crossfade_ms":    crossfade_ms,
            "mode":            mode,
        }

        # --- Update and re-save manifest ---
        music_result["stitched"] = stitched_block

        save_json(
            data=music_result,
            file_name="manifest",
            directory=str(output_dir),
            overwrite=True,
        )
        log(f"Manifest updated → {output_dir}/manifest.json")

        return music_result

    # -------------------------------------------------------------------------
    # Internal — segment stitching logic
    # -------------------------------------------------------------------------

    def _stitch_segments(
        self,
        segments: list,
        mode: str,
        gap_ms: int,
        crossfade_ms: int,
    ) -> AudioSegment:
        """
        Combine a list of AudioSegment objects according to the chosen mode.

        Args:
            segments     (list):  Ordered list of AudioSegment objects.
            mode         (str):   "crossfade" | "gap" | "direct"
            gap_ms       (int):   Silence duration in ms (used when mode="gap").
            crossfade_ms (int):   Crossfade overlap in ms (used when mode="crossfade").

        Returns:
            AudioSegment: The fully stitched audio segment.
        """
        if mode == "crossfade":
            result = segments[0]
            for seg in segments[1:]:
                result = result.append(seg, crossfade=crossfade_ms)
            return result

        elif mode == "gap":
            silence = AudioSegment.silent(duration=gap_ms)
            result  = segments[0]
            for seg in segments[1:]:
                result = result + silence + seg
            return result

        else:  # direct
            result = segments[0]
            for seg in segments[1:]:
                result = result + seg
            return result

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def display_stitch(self, manifest: dict) -> None:
        """
        Pretty-print the stitched result block from an updated manifest.

        Args:
            manifest (dict): Manifest returned by stitch() — must contain
                             a 'stitched' key.
        """
        stitched = manifest.get("stitched")
        if not stitched:
            console.print("[bold red]No 'stitched' block found in manifest.[/bold red]")
            return

        mode         = stitched.get("mode", "—")
        included     = stitched.get("tracks_included", 0)
        skipped      = stitched.get("tracks_skipped", 0)
        gap_ms       = stitched.get("gap_ms", 0)
        crossfade_ms = stitched.get("crossfade_ms", 0)
        file_path    = stitched.get("file_path", "—")

        # Mode badge
        mode_colours = {
            "crossfade": "[bold magenta]crossfade[/bold magenta]",
            "gap":       "[bold yellow]gap[/bold yellow]",
            "direct":    "[bold white]direct[/bold white]",
        }
        mode_badge = mode_colours.get(mode, mode)

        # Parameter line
        if mode == "crossfade":
            param_line = f"[dim]Crossfade :[/dim]  [bold magenta]{crossfade_ms}ms[/bold magenta]"
        elif mode == "gap":
            param_line = f"[dim]Gap       :[/dim]  [bold yellow]{gap_ms}ms[/bold yellow]"
        else:
            param_line = "[dim]Parameters:[/dim]  [dim]none[/dim]"

        lines = [
            f"[dim]Mode      :[/dim]  {mode_badge}",
            param_line,
            f"[dim]Included  :[/dim]  [bold green]{included}[/bold green] track(s)",
            f"[dim]Skipped   :[/dim]  [bold red]{skipped}[/bold red] track(s)",
            f"[dim]Output    :[/dim]  [dim italic]{file_path}[/dim italic]",
        ]

        console.print(
            Panel(
                Text.from_markup("\n".join(lines)),
                title="[bold white] 🎛  Mix Maker — Stitched Output [/bold white]",
                border_style="magenta",
                padding=(1, 2),
            )
        )
        console.print()

    def stitch_and_display(
        self,
        music_result: dict,
        gap_ms: int = 0,
        crossfade_ms: int = 0,
    ) -> dict:
        """
        Convenience method — stitch and immediately display the result.

        Args:
            music_result  (dict): Output from MusicGenerator.generate().
            gap_ms        (int):  Silence gap between tracks in ms. Default: 0.
            crossfade_ms  (int):  Crossfade overlap in ms. Default: 0.

        Returns:
            dict: Updated manifest with stitched block.
        """
        result = self.stitch(music_result, gap_ms=gap_ms, crossfade_ms=crossfade_ms)
        self.display_stitch(result)
        return result
