import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0"))

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_data")
DATA_DIR = os.getenv("DATA_DIR", "./data")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
