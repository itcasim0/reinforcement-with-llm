import os

from openai import OpenAI
from dotenv import load_dotenv

from config.paths import ROOT_DIR

load_dotenv(ROOT_DIR / ".env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
