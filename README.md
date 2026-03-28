# Clappix RAG Bot

AI-powered Telegram bot that answers questions based on your company's knowledge base using RAG (Retrieval-Augmented Generation).

## Architecture

```
User Question
     │
     ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Telegram    │───▶│  RAG Engine  │───▶│   ChromaDB  │
│  Bot API     │    │              │    │  (vectors)  │
└─────────────┘    │  1. Embed    │    └─────────────┘
                   │  2. Search   │           │
                   │  3. Context  │◀──────────┘
                   └──────┬───────┘    top-3 chunks
                          │
                          ▼
                   ┌──────────────┐
                   │   OpenAI     │
                   │   GPT-4o     │
                   │  + context   │
                   └──────┬───────┘
                          │
                          ▼
                   Answer to User
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Bot Framework | python-telegram-bot |
| Vector DB | ChromaDB |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | OpenAI GPT-4o |
| API | FastAPI (health checks) |
| Deployment | Docker Compose |

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/YourUser/clappix-rag-bot.git
cd clappix-rag-bot
cp .env.example .env
# Edit .env with your API keys
```

### 2. Add Your Documents

Place `.md` or `.txt` files in the `data/` directory. The bot will automatically index them on startup.

### 3. Run

```bash
docker compose up --build
```

The bot will:
1. Load and split documents into chunks
2. Create embeddings and store in ChromaDB
3. Start the Telegram bot
4. Listen for questions

### 4. Talk to Your Bot

Open your bot in Telegram and ask questions about your documents.

## Commands

| Command | Description |
|---------|------------|
| `/start` | Welcome message |
| `/help` | Show usage examples |
| `/reload` | Reload knowledge base (admin only) |

## How RAG Works

1. **Document Loading** — Files from `data/` are split into overlapping chunks (~500 words each)
2. **Embedding** — Each chunk is converted to a vector using OpenAI's embedding model
3. **Storage** — Vectors are stored in ChromaDB for fast similarity search
4. **Query** — When a user asks a question, it's also embedded and the 3 most similar chunks are retrieved
5. **Answer** — The retrieved chunks + user question are sent to GPT-4o, which generates a grounded answer

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reload` | POST | Reload documents |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | Chat model (default: gpt-4o) |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token from @BotFather |
| `ADMIN_USER_ID` | Telegram user ID for admin commands |

## License

MIT
