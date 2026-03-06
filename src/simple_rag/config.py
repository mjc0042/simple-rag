import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini")  # Model name (e.g., gpt-4o-mini, deepseek-exp, claude-sonnet-4-20250514)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")  # HuggingFace embedding model
HF_TOKEN = os.getenv("HF_TOKEN", None)