# trial.py
from dotenv import load_dotenv
import os

from src.provider.llm import LLM
from src.provider.music_generator import MusicGenerator
from src.core.mix_maker import MixMaker

load_dotenv()

llm       = LLM(api_key=os.getenv("LLM_KEY"))
music_gen = MusicGenerator(api_key=os.getenv("GOOGLE_API_KEY"))
mix_maker = MixMaker()

user_prompt = "Epic cinematic battle song with deep bass and raging drums"
style_id    = "indian_classical"

llm_result   = llm.generate_variations(user_prompt, style_id, n=3)
music_result = music_gen.generate_and_display(llm_result)
mix_maker.stitch_and_display(music_result)