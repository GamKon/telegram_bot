# Disable tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

from dotenv import load_dotenv
import logging

from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes

#from vit_base_patch16_224 import image_category_16_224
from vit_base_patch32_384 import image_category_32_384
from microsoft_phi_1_5 import chat_phi_1_5
from facebook_wmt19 import facebook_wmt19_en_ru, facebook_wmt19_ru_en
from runwayml_stable_diffusion_v1_5 import stable_diffusion_v1_5
from openai_whisper_large_v3 import openai_whisper_large_v3

# Load environment variables form .env
load_dotenv()
TELEGRAM_BOT_TOKEN=os.getenv('TELEGRAM_BOT_TOKEN')

# Set up logging module to monitor Telegram
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Dir to store media
custom_path     = "media"


# /start and /help commans
help_message    =   "I'm an AI bot. Please send me:\n\
                    - voice message to translate and transcript\n\
                    - picture to categirize.\n\
                    Commands:\n\
                    /img _description_ to generate a photo using runwayml/stable-diffusion-v1-5\n\
                    /txt _description_ to generate a text with microsoft/phi-1_5\n\
                    /imgr and /txtr for russian _description_"
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=help_message)


# /img command - Image generator
async def image_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    description = " ".join(context.args)
    generated_picture = stable_diffusion_v1_5(description, "generated_images")
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=generated_picture)

# /imgr command - Russian description Image Generator
# TODO merge these two defs into one. Use conditionals for en_ru / ru_en
async def image_ru_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    description = facebook_wmt19_ru_en(str(" ".join(context.args)))
    generated_picture = stable_diffusion_v1_5(description, "generated_images")
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=generated_picture)


# /txt command - Text Generator + Translator
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_bot_ru_answer = "\n_________\nПеревод:\n_________\n"
    chat_bot_en_answer = chat_phi_1_5(" ".join(context.args))
    # Cut excessive text
    chat_bot_en_answer = str(chat_bot_en_answer.partition("<|endoftext")[0])
    # Split text to translate by 1024 symbols
    # TODO: make split by the end of sentence, not just by num of chars
    answers_en = [chat_bot_en_answer[i:i + 1024] for i in range(0, len(chat_bot_en_answer), 1024)]
    for answer_en in answers_en:
        chat_bot_ru_answer = chat_bot_ru_answer + facebook_wmt19_en_ru(answer_en)
    chat_bot_answer = chat_bot_en_answer + chat_bot_ru_answer
    # Split output by 4096 symbols
    answers = [chat_bot_answer[i:i + 4096] for i in range(0, len(chat_bot_answer), 4096)]
    for answer in answers:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

# /txtr command - russian only Text Generator + Translator
async def echo_ru(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_bot_ru_answer = ""
    description = facebook_wmt19_ru_en(" ".join(context.args))
    chat_bot_en_answer = chat_phi_1_5(description)
    # Cut excessive text
    chat_bot_en_answer = str(chat_bot_en_answer.partition("<|endoftext")[0])
    # Split text to translate by 1024 symbols
    # TODO: make split by the end of sentence, not just by num of chars
    answers_en = [chat_bot_en_answer[i:i + 1024] for i in range(0, len(chat_bot_en_answer), 1024)]
    for answer_en in answers_en:
        chat_bot_ru_answer = chat_bot_ru_answer + facebook_wmt19_en_ru(answer_en)
    # Split output by 4096 symbols
    answers = [chat_bot_ru_answer[i:i + 4096] for i in range(0, len(chat_bot_ru_answer), 4096)]
    for answer in answers:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


# Image classificator
async def photo_classification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    new_file    = await update.message.effective_attachment[-1].get_file()
    file_name   = new_file.file_path.split("/")[-1]
    # print("!!!!!!!-"+new_file.file_path.split("/")[-1]+"-!!!!!!!")
    file_path   = await new_file.download_to_drive(custom_path="images/"+file_name)
    en_text     = "Most likely it's " + image_category_32_384(file_path)
    ru_text     = facebook_wmt19_en_ru(en_text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=en_text + "\n" + ru_text)

# Audio processing
async def voice_trans(update: Update, context: ContextTypes.DEFAULT_TYPE):
    new_file    = await update.message.effective_attachment.get_file()
    file_name   = new_file.file_path.split("/")[-1]
    file_path   = await new_file.download_to_drive(custom_path="voice/"+file_name)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=openai_whisper_large_v3(file_path))

# Error message
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.\n" + help_message)

if __name__ == '__main__':
    application     = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    start_handler   = CommandHandler('start', help_command)
    application.add_handler(start_handler)
    help_handler    = CommandHandler('help', help_command)
    application.add_handler(help_handler)

    img_handler     = CommandHandler('img', image_generation)
    application.add_handler(img_handler)

    img_ru_handler  = CommandHandler('imgr', image_ru_generation)
    application.add_handler(img_ru_handler)

    echo_handler    = CommandHandler('txt', echo)
    application.add_handler(echo_handler)
    echo_ru_handler    = CommandHandler('txtr', echo_ru)
    application.add_handler(echo_ru_handler)

    image_handler   = MessageHandler(filters.PHOTO , photo_classification)
    application.add_handler(image_handler)

    audio_handler   = MessageHandler(filters.VOICE , voice_trans)
    application.add_handler(audio_handler)

    # Other handlers
    unknown_handler = MessageHandler(filters.COMMAND, error)
    application.add_handler(unknown_handler)

    application.run_polling()
