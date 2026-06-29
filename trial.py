import os
from src.provider.llm import LLM
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = LLM(api_key)

prompt = ("Deep lord of the rings inspired music")
n = 5
style_id = "ambient"

response = llm.generate_variations(prompt, style_id)
