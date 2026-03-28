import logging
import asyncio
import threading
import uvicorn
from fastapi import FastAPI
from app.config import TELEGRAM_BOT_TOKEN, OPENAI_API_KEY
from app import rag
from app.bot import create_bot

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# FastAPI for health checks
api = FastAPI(title="Clappix RAG Bot")


@api.get("/health")
async def health():
    return {"status": "ok"}


@api.post("/reload")
async def reload_docs():
    count = rag.load_documents()
    return {"status": "ok", "chunks": count}


def run_api():
    uvicorn.run(api, host="0.0.0.0", port=8080, log_level="warning")


def main():
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY is not set")
        return
    if not TELEGRAM_BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN is not set")
        return

    # Load documents on startup
    log.info("Loading documents into ChromaDB...")
    count = rag.load_documents()
    log.info("Indexed %d chunks", count)

    # Start FastAPI in a background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    log.info("FastAPI health endpoint running on :8080")

    # Start Telegram bot (blocking)
    log.info("Starting Telegram bot...")
    bot = create_bot()
    bot.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
