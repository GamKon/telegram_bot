# Disable tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

from dotenv import load_dotenv
import logging

from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes

from vit_base_patch16_224 import image_category_16_224
from vit_base_patch32_384 import image_category_32_384
from microsoft_phi_1_5 import chat_phi_1_5
from facebook_wmt19_en_ru import facebook_wmt19_en_ru

load_dotenv()
TELEGRAM_BOT_TOKEN=os.getenv('TELEGRAM_BOT_TOKEN')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm an AI, please talk to me or send me a picture!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_bot_ru_answer = "\n-------------------------------\nRussian translation:\n"
    chat_bot_en_answer = chat_phi_1_5(update.message.text)
    # Cut excessive text
    chat_bot_en_answer = str(chat_bot_en_answer.partition("<|endoftext")[0])
    # Split text to translate by 4096 symbols
    # Ideally todo: make split by the end of sentance, not just by num of chars
    answers_en = [chat_bot_en_answer[i:i + 768] for i in range(0, len(chat_bot_en_answer), 768)]
    for answer_en in answers_en:
        chat_bot_ru_answer = chat_bot_ru_answer + str(facebook_wmt19_en_ru(str(answer_en)))
    chat_bot_answer = str(chat_bot_en_answer + chat_bot_ru_answer)
    # Split output by 4096 symbols
    answers = [chat_bot_answer[i:i + 4096] for i in range(0, len(chat_bot_answer), 4096)]
    for answer in answers:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    new_file    = await update.message.effective_attachment[-1].get_file()
    file_path   = await new_file.download_to_drive()
    en_text     = "Most likely it's " + image_category_32_384(file_path)
    ru_text     = facebook_wmt19_en_ru(en_text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=en_text + "\n" + ru_text)

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