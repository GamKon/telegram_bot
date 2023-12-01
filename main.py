import os
from dotenv import load_dotenv
import logging

from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes

from vit_base_patch16_224 import image_category

load_dotenv()
TELEGRAM_BOT_TOKEN=os.getenv('TELEGRAM_BOT_TOKEN')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    new_file = await update.message.effective_attachment[-1].get_file()
    file_path = await new_file.download_to_drive()

    await context.bot.send_message(chat_id=update.effective_chat.id, text=image_category(file_path))

    #print(file_path)

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")

if __name__ == '__main__':
    application     = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    start_handler   = CommandHandler('start', start)
    application.add_handler(start_handler)

    echo_handler    = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    application.add_handler(echo_handler)

    image_handler   = MessageHandler(filters.PHOTO , image)
    application.add_handler(image_handler)

    # Other handlers
    unknown_handler = MessageHandler(filters.COMMAND, unknown)
    application.add_handler(unknown_handler)

    application.run_polling()