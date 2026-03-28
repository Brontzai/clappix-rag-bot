import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from app.config import TELEGRAM_BOT_TOKEN, ADMIN_USER_ID
from app import rag

log = logging.getLogger(__name__)


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I'm the Clappix AI assistant.\n\n"
        "Ask me anything about Clappix products, services, or pricing.\n\n"
        "Commands:\n"
        "/help — show available commands\n"
        "/reload — reload knowledge base (admin only)"
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Just send me a question and I'll answer based on the Clappix knowledge base.\n\n"
        "Examples:\n"
        "• What is Clappix?\n"
        "• What products do you offer?\n"
        "• How does AI Landing Agent work?\n"
        "• What are the pricing options?"
    )


async def cmd_reload(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if ADMIN_USER_ID and user_id != ADMIN_USER_ID:
        await update.message.reply_text("This command is for admins only.")
        return

    await update.message.reply_text("Reloading knowledge base...")
    count = rag.load_documents()
    await update.message.reply_text(f"Done! Indexed {count} chunks.")


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    if not query:
        return

    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        response = rag.answer(query)
        await update.message.reply_text(response)
    except Exception as e:
        log.error("Error answering query: %s", e, exc_info=True)
        await update.message.reply_text(
            "Sorry, something went wrong. Please try again later."
        )


def create_bot() -> Application:
    """Create and configure the Telegram bot application."""
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app
