#!/usr/bin/env python

import logging
import os

from flask import Flask, request
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from dotenv import load_dotenv
from utils.prepare_data import prepare_image
from tensorflow.keras.models import load_model

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
WEBHOOK_PORT = os.getenv("WEBHOOK_PORT")

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

mask_model = load_model("happy_sad_model.keras")

# Initialize Flask app
app = Flask(__name__)

# Initialize Telegram application
application = Application.builder().token(TELEGRAM_TOKEN).build()


# Define a few command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Привет, {user.mention_html()}! Я бот, который может"
        + " распознать, грустного или веселого человека.",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Отправьте картинку, чтобы я смог сказать, грустный или веселый это человек."
    )


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo_file = await update.message.photo[-1].get_file()
    await photo_file.download_to_drive("user_image.jpg")

    image_for_prediction = prepare_image("user_image.jpg")
    prediction = mask_model.predict(image_for_prediction)
    logger.info(f"prediction probability: {prediction[0][0]}")
    result = (
        "Кажется, что счастливый 😊"
        if prediction[0][0] < 0.74
        else "Кажется, что грустный 😔"
    )

    os.remove("user_image.jpg")
    await update.message.reply_text(f"{result}\nPrediction: {prediction[0][0]}")


application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("help", help_command))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
application.add_handler(MessageHandler(filters.PHOTO, handle_image))


@app.route("/webhook", methods=["POST"])
def webhook():
    if request.method == "POST":
        update = Update.de_json(request.get_json(force=True), application.bot)
        application.update_queue.put_nowait(update)
        return "OK"
    return "Method not allowed", 405


async def set_webhook():
    await application.bot.set_webhook(WEBHOOK_URL)


if __name__ == "__main__":
    import asyncio

    asyncio.run(set_webhook())
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
