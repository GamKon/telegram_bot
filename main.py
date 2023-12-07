# # Disable tensorflow warnings
import os
import json
import shutil
from time import sleep

# from dotenv import load_dotenv
# import logging

from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes

# from vit_base_patch16_224 import image_category_16_224
from models.vit_base_patch32_384 import image_category_32_384
from models.microsoft_phi_1_5 import chat_phi_1_5
from models.facebook_wmt19 import facebook_wmt19_en_ru, facebook_wmt19_ru_en
from models.runwayml_stable_diffusion_v1_5 import stable_diffusion_v1_5
from models.openai_whisper_large_v3 import openai_whisper_large_v3
from models.stabilityai_stable_diffusion_xl_base_1_0 import stable_diffusion_xl_base_1_0, stable_diffusion_xl_base_refiner_1_0
# from models.stabilityai_sd_x2_latent_upscaler import sd_x2_latent_upscaler
from models.stabilityai_stable_diffusion_x4_upscaler import stable_diffusion_x4_upscaler
from models.TheBloke_Llama_2_13B_Chat_GPTQ import Llama_2_13B_chat_GPTQ
from models.philschmid_bart_large_cnn_samsum import bart_large_cnn_samsum

# split to sentances
from split_into_sentences import split_into_sentences

# Needs nltk
#from nltk import tokenize
# nltk.download('punkt')

global GLOBAL_GPT_CONTEXT
GLOBAL_GPT_CONTEXT = " "


def summarize_context(chat_history):
    print("\n--------------------------------------------\n")
    sentences_list = split_into_sentences(chat_history)
    print(sentences_list)
    print("\n--------------------------------------------\n")
    unic_sentences_list = list(dict.fromkeys(sentences_list))
    print(unic_sentences_list)
    print("\n--------------------------------------------\n")
    return unic_sentences_list

# -----------------------------------------------------------------------------------------
# Command handlers
# -----------------------------------------------------------------------------------------

# /start and /help commans
# Reset GLOBAL_GPT_CONTEXT and print help_message
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global GLOBAL_GPT_CONTEXT
    GLOBAL_GPT_CONTEXT = os.getenv('GLOBAL_GPT_CONTEXT')
    await context.bot.send_message(chat_id=update.effective_chat.id, text=HELP_MESSAGE)

# /img command - Image generator
# stable_diffusion_xl_base_1_0
async def image_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        num_inference_steps = int(context.args[0])
        context.args.pop(0)
    except ValueError:
        num_inference_steps = 50
    description = " ".join(context.args)

    # description = facebook_wmt19_ru_en(description)
    # debug_print("description after ru_en translation: "+description)
    generated_picture = stable_diffusion_xl_base_1_0(description, "generated_images", num_inference_steps)
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=generated_picture, caption=description)
    # Experiment to see picture after every iteration
    # for iteration in range(1, num_inference_steps):
    #     generated_picture = stable_diffusion_xl_base_1_0(description, "generated_images", iteration)
    #     await context.bot.send_photo(chat_id=update.effective_chat.id, photo=generated_picture, caption=str(iteration)+" -- "+description)

# /imgh command - Image generator using Refiner
# stable_diffusion_xl_base_refiner_1_0
async def image_refine_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(context.args)
    try:
        num_inference_steps = int(context.args[0])
        context.args.pop(0)
    except ValueError:
        num_inference_steps = 20

    description = " ".join(context.args)
    # debug_print("description: "+description)
    # description = facebook_wmt19_ru_en(description)
    # debug_print("description after ru_en translation: "+description)
    # Base + Refiner. Very slow ~20 min.
    generated_picture = stable_diffusion_xl_base_refiner_1_0(description, "generated_images", num_inference_steps)
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=generated_picture, caption=description)

# /imgr command - Russian description Image Generator
# runwayml_stable_diffusion_v1_5
async def image_ru_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    description = " ".join(context.args)

    # description = facebook_wmt19_ru_en(description)
    # debug_print("description after ru_en translation: "+description)
    #generated_picture = stable_diffusion_xl_base_1_0(description, "generated_images")
    generated_picture = stable_diffusion_v1_5(description, "generated_images")
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=generated_picture, caption=description)

# /t command - Chat
async def gpt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global GLOBAL_GPT_CONTEXT
    user_message = str(" ".join(context.args))

#   Handle crash happening most likely, not enought RAM
    for i in range(1, 10):
        try:
            gpt_answer = Llama_2_13B_chat_GPTQ(user_message, GLOBAL_GPT_CONTEXT)
            break
        except ValueError:
            print("!!!!!!!!!!!!!!!!!!!-------------CRASH-"+str(i)+"------------!!!!!!!!!!!!!!!!!!!")
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Crash, retrying "+str(i))
            sleep(20)
            #gpt_answer = Llama_2_13B_chat_GPTQ(user_message, GLOBAL_GPT_CONTEXT)
    debug_print(gpt_answer)
    print(gpt_answer)
    gpt_answer = str(gpt_answer.partition("[/INST]")[2])
#    debug_print("gpt_answer: "+gpt_answer)
    # Split output by 4096 symbols

    answers = [gpt_answer[i:i + 4096] for i in range(0, len(gpt_answer), 4096)]
    for answer in answers:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)
#    GLOBAL_GPT_CONTEXT = GLOBAL_GPT_CONTEXT.split("[/INST]")[0]
    GLOBAL_GPT_CONTEXT = str(GLOBAL_GPT_CONTEXT.split("[/INST]")[0]) + " " + user_message + gpt_answer
    GLOBAL_GPT_CONTEXT = " ".join(summarize_context(GLOBAL_GPT_CONTEXT))
#    debug_print("New GLOBAL_GPT_CONTEXT: "+GLOBAL_GPT_CONTEXT)

# /tr command - Chat via russian Translator
async def gpt_ru(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_bot_ru_answer = ""
    description = facebook_wmt19_ru_en(" ".join(context.args))
#    debug_print(description)
    chat_bot_en_answer = Llama_2_13B_chat_GPTQ(description)
    # Cut excessive text
    chat_bot_en_answer = str(chat_bot_en_answer.partition("[/INST]")[2])
#    debug_print(chat_bot_en_answer)
    # Split text to translate by 1024 symbols
    # TODO: make split by the end of sentence, not just by num of chars
#    answers_en = [chat_bot_en_answer[i:i + 2048] for i in range(0, len(chat_bot_en_answer), 2048)]
#    for answer_en in answers_en:
    chat_bot_ru_answer = chat_bot_ru_answer + facebook_wmt19_en_ru(chat_bot_en_answer)
    # Split output by 4096 symbols
#    debug_print(chat_bot_ru_answer)
    answers = [chat_bot_ru_answer[i:i + 4096] for i in range(0, len(chat_bot_ru_answer), 4096)]
    for answer in answers:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

# /cont command - sets start context from input
async def new_gpt_context(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global GLOBAL_GPT_CONTEXT
    GLOBAL_GPT_CONTEXT = str(" ".join(context.args))
    await context.bot.send_message(chat_id=update.effective_chat.id, text="New context is:\n"+GLOBAL_GPT_CONTEXT)


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

# ---------------
# Summarize with bart_large_cnn_samsum
# ---------------
async def summarize_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = str(" ".join(context.args))
#    debug_print("user_message to summarize: "+user_message)
    summary = bart_large_cnn_samsum(user_message)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Summary of message:\n"+summary)

async def summarize_gpt_context(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global GLOBAL_GPT_CONTEXT
    GLOBAL_GPT_CONTEXT = bart_large_cnn_samsum(GLOBAL_GPT_CONTEXT)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Summary of chat history:\n"+GLOBAL_GPT_CONTEXT)


# Image classificator and 2x Upscaler
async def photo_classification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    new_file    = await update.message.effective_attachment[-1].get_file()
    file_name   = new_file.file_path.split("/")[-1]
    file_path   = await new_file.download_to_drive(custom_path="images/"+file_name)
    description = image_category_32_384(file_path)
    en_text     = "Most likely it's " + description
    ru_text     = facebook_wmt19_en_ru(en_text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=en_text + "\n" + ru_text)
    # Upscaler
    upscaled_picture = stable_diffusion_x4_upscaler(description, str(file_path))
#    debug_print(upscaled_picture)
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=upscaled_picture)

# Audio processing
async def voice_trans(update: Update, context: ContextTypes.DEFAULT_TYPE):
    new_file    = await update.message.effective_attachment.get_file()
    file_name   = new_file.file_path.split("/")[-1]
    while True:
        try:
            file_path   = await new_file.download_to_drive(custom_path="voice/"+file_name)
            break
        except shutil.SameFileError:
            # TODO parse file_name for better making unic name
            file_name = "1" + file_name
    await context.bot.send_message(chat_id=update.effective_chat.id, text=openai_whisper_large_v3(file_path))

# Error message
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.\n" + HELP_MESSAGE)

def debug_print(to_print):
    print("\n!!!------------------------\n"+to_print+"\n------------------------!!!\n")


if __name__ == '__main__':

# Variables ------------------------------------------------------------------------------
    HELP_MESSAGE        = "".join(json.loads(os.getenv('HELP_MESSAGE')))
    TELEGRAM_BOT_TOKEN  = os.getenv('TELEGRAM_BOT_TOKEN')

    # Set up logging module to monitor Telegram
    # logging.basicConfig(
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #     level=logging.INFO
    # )

    application     = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    start_handler   = CommandHandler('start', help_command)
    application.add_handler(start_handler)
    help_handler    = CommandHandler('help', help_command)
    application.add_handler(help_handler)

    # img_fooocus_handler  = CommandHandler('i', image_fooocus_generation)
    # application.add_handler(img_fooocus_handler)
    img_handler     = CommandHandler('img', image_generation)
    application.add_handler(img_handler)
    imgh_handler    = CommandHandler('imgh', image_refine_generation)
    application.add_handler(imgh_handler)
    img_ru_handler  = CommandHandler('imgr', image_ru_generation)
    application.add_handler(img_ru_handler)

    gpt_handler     = CommandHandler('t', gpt)
    application.add_handler(gpt_handler)
    gpt_ru_handler  = CommandHandler('tr', gpt_ru)
    application.add_handler(gpt_ru_handler)

    echo_handler    = CommandHandler('txt', echo)
    application.add_handler(echo_handler)
    echo_ru_handler = CommandHandler('txtr', echo_ru)
    application.add_handler(echo_ru_handler)

    image_handler   = MessageHandler(filters.PHOTO , photo_classification)
    application.add_handler(image_handler)

    audio_handler   = MessageHandler(filters.VOICE , voice_trans)
    application.add_handler(audio_handler)

    new_context_handler         = CommandHandler('cont', new_gpt_context)
    application.add_handler(new_context_handler)

    summarize_message_handler   = CommandHandler('s',summarize_message)
    application.add_handler(summarize_message_handler)
    summarize_context_handler   = CommandHandler('summ',summarize_gpt_context)
    application.add_handler(summarize_context_handler)

    # Other handlers
    unknown_handler = MessageHandler(filters.COMMAND, error)
    application.add_handler(unknown_handler)

    application.run_polling()
