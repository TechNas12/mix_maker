from src.utils.load_json import load_json
from src.core.mix_maker import MixMaker

music_result = load_json("artifacts/2026-05-31_18-37-29/manifest.json")

mix_maker = MixMaker()
mix_maker.stitch_and_display(music_result)