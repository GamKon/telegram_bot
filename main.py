# # Disable tensorflow warnings
import os
import json
import shutil
import emoji
from time import sleep

from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes, PicklePersistence

# from vit_base_patch16_224 import image_category_16_224
from models.vit_base_patch32_384 import image_category_32_384
from models.microsoft_phi_1_5 import chat_phi_1_5
from models.facebook_wmt19 import facebook_wmt19_en_ru, facebook_wmt19_ru_en
from models.runwayml_stable_diffusion_v1_5 import stable_diffusion_v1_5
from models.openai_whisper_large_v3 import openai_whisper_large_v3
from models.stabilityai_stable_diffusion_xl_base_1_0 import stable_diffusion_xl_base_1_0, stable_diffusion_xl_base_refiner_1_0
from models.stabilityai_stable_diffusion_x4_upscaler import stable_diffusion_x4_upscaler
from models.TheBloke_Llama_2_13B_Chat_GPTQ import Llama_2_13B_chat_GPTQ
from models.meta_llama_Llama_2_13b_chat_hf import Llama_2_13b_chat_hf
from models.philschmid_bart_large_cnn_samsum import bart_large_cnn_samsum

# -----------------------------------------------------------------------------------------
# Command handlers
# -----------------------------------------------------------------------------------------
# /start and /help commans
# Print HELP_MESSAGE
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=HELP_MESSAGE)

# -----------------------------------------------------------------------------------------
# Image generators
# -----------------------------------------------------------------------------------------
# /img command - Image generator
# stable_diffusion_xl_base_1_0
async def image_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
# If first argument is number, use it as num_inference_steps
    try:
        num_inference_steps = int(context.args[0])
        context.args.pop(0)
    except ValueError:
        num_inference_steps = 50

    description = str(" ".join(context.args)).strip()

    generated_picture = stable_diffusion_xl_base_1_0(description, "data/generated_images", num_inference_steps)
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=generated_picture, caption=description)

    # Experiment to see picture after each iteration
        # for iteration in range(1, num_inference_steps):
        #     generated_picture = stable_diffusion_xl_base_1_0(description, "data/generated_images", iteration)
        #     await context.bot.send_photo(chat_id=update.effective_chat.id, photo=generated_picture, caption=str(iteration)+" -- "+description)

# -----------------------------------------------------------------------------------------
# /imgh command - Image generator with added Refiner
# stable_diffusion_xl_base_refiner_1_0
async def image_refine_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
# If first argument is number, use it as num_inference_steps
    try:
        num_inference_steps = int(context.args[0])
        context.args.pop(0)
    except ValueError:
        num_inference_steps = 20

    description = str(" ".join(context.args)).strip()

    generated_picture = stable_diffusion_xl_base_refiner_1_0(description, "data/generated_images", num_inference_steps)
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=generated_picture, caption=description)

# -----------------------------------------------------------------------------------------
# /imgr command - Russian description Image Generator
# runwayml_stable_diffusion_v1_5
async def image_ru_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
# If first argument is number, use it as num_inference_steps
    try:
        num_inference_steps = int(context.args[0])
        context.args.pop(0)
    except ValueError:
        num_inference_steps = 50

    description_ru = str(" ".join(context.args)).strip()
    description = facebook_wmt19_ru_en(description_ru)

    generated_picture = stable_diffusion_xl_base_1_0(description, "data/generated_images", num_inference_steps)
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=generated_picture, caption=description_ru+"\n"+description_ru)


# -----------------------------------------------------------------------------------------
# Chat
# -----------------------------------------------------------------------------------------
# /tm command - Chat
async def meta_llama_2(update: Update, context: ContextTypes.DEFAULT_TYPE):
#    return
# How deep to summarize
    summarize_depth = 4
# Get question, stripping double spaces
    user_prompt = str(" ".join(context.args)).strip()
# Construct context from history
    if "chat_history" not in context.user_data.keys(): context.user_data["chat_history"] = []
    context_string = construct_context_string_from_history(context.user_data["chat_history"])

# Cut old history if whole string to sent is longer than

 #!   # while len((user_prompt+context_string+str(os.getenv('INITIAL_PROMPT'))).split())+4 > 2048:
    #     context.user_data["chat_history"].pop(0)
    #     debug_print("POP oldest record")
    #     await context.bot.send_message(chat_id=update.effective_chat.id, text="POP oldest record")
    #     context_string = construct_context_string_from_history(context.user_data["chat_history"])

    # From documentation: To increase temp_state buffer:
    # from auto_gptq import exllama_set_max_input_length
    # model = exllama_set_max_input_length(model, max_input_length=xxxx)

# Ask Llama
    # 10 tries to handle crashes. Most likely, because out of RAM
    for i in range(1, 10):
        try:
#            gpt_answer = Llama_2_13B_chat_GPTQ(user_prompt, context_string, str(os.getenv('INITIAL_PROMPT')))
            gpt_answer = Llama_2_13b_chat_hf(user_prompt, context_string, str(os.getenv('INITIAL_PROMPT')))
            break
        except ValueError:
            print("!!!!!!!!!!!!!!!!!!!-------------CRASH-"+str(i)+"------------!!!!!!!!!!!!!!!!!!!")
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Crash, retrying "+str(i))
            sleep(5)

# Remove initial context from answer
    gpt_answer = str(gpt_answer.split("[/INST]")[-1])

    # Split output by 4096 symbols
    # answers = [gpt_answer[i:i + 4096] for i in range(len(gpt_answer), 4096)]
    # for answer in answers:
# Sent answer to the chat
    await context.bot.send_message(chat_id=update.effective_chat.id, text=str(len(context.user_data["chat_history"]))+" "+gpt_answer)

# Save question and answer to list of dictionaries
    context.user_data["chat_history"].append({"question": emoji.replace_emoji(user_prompt, replace=''), "answer": emoji.replace_emoji(gpt_answer.strip(), replace='')})

#    debug_print("Context AFTER the answer - debug_context_string:\n"+str(context.user_data["chat_history"]))
    debug_print("Context !AFTER! the answer: "+str(len(context.user_data["chat_history"]))+"\n"+json.dumps(context.user_data["chat_history"], indent=4))

#!! # Summarize [summarize_depth] answer
#     if len(context.user_data["chat_history"]) >= summarize_depth and len(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']) > 100 :
#         summarized_answer = bart_large_cnn_samsum(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']))
#         debug_print("BEFORE summarizing: "+str(len(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer'])))+"\n"+str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']))
#         await context.bot.send_message(chat_id=update.effective_chat.id, text=str(len(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']))))
#         debug_print("AFTER summarizing:\n"+summarized_answer)
#         context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer'] = summarized_answer
#         debug_print("Context AFTER Summarizing: "+str(len(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer'])))+"\n"+json.dumps(context.user_data["chat_history"], indent=4))
#         await context.bot.send_message(chat_id=update.effective_chat.id, text=str(len(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']))))

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# /t command - Chat
async def tb_llama_2(update: Update, context: ContextTypes.DEFAULT_TYPE):
# How deep to summarize
    summarize_depth = 4
# Get question, stripping double spaces
    user_prompt = str(" ".join(context.args)).strip()
# Construct context from history
    if "chat_history" not in context.user_data.keys(): context.user_data["chat_history"] = []
    context_string = construct_context_string_from_history(context.user_data["chat_history"])

# Cut old history if whole string to sent is longer than

 #!   # while len((user_prompt+context_string+str(os.getenv('INITIAL_PROMPT'))).split())+4 > 2048:
    #     context.user_data["chat_history"].pop(0)
    #     debug_print("POP oldest record")
    #     await context.bot.send_message(chat_id=update.effective_chat.id, text="POP oldest record")
    #     context_string = construct_context_string_from_history(context.user_data["chat_history"])

    # From documentation: To increase temp_state buffer:
    # from auto_gptq import exllama_set_max_input_length
    # model = exllama_set_max_input_length(model, max_input_length=xxxx)

# Ask Llama
    # 10 tries to handle crashes. Most likely, because out of RAM
    for i in range(1, 10):
        try:
#            gpt_answer = Llama_2_13B_chat_GPTQ(user_prompt, context_string, str(os.getenv('INITIAL_PROMPT')))
            gpt_answer = Llama_2_13B_chat_GPTQ(user_prompt, context_string, str(os.getenv('INITIAL_PROMPT')))
            break
        except ValueError:
            print("!!!!!!!!!!!!!!!!!!!-------------CRASH-"+str(i)+"------------!!!!!!!!!!!!!!!!!!!")
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Crash, retrying "+str(i))
            sleep(5)

# Remove initial context from answer
    gpt_answer = str(gpt_answer.split("[/INST]")[-1])

    # Split output by 4096 symbols
    # answers = [gpt_answer[i:i + 4096] for i in range(len(gpt_answer), 4096)]
    # for answer in answers:
# Sent answer to the chat
    await context.bot.send_message(chat_id=update.effective_chat.id, text=str(len(context.user_data["chat_history"]))+" "+gpt_answer)

# Save question and answer to list of dictionaries
    context.user_data["chat_history"].append({"question": emoji.replace_emoji(user_prompt, replace=''), "answer": emoji.replace_emoji(gpt_answer.strip(), replace='')})

#    debug_print("Context AFTER the answer - debug_context_string:\n"+str(context.user_data["chat_history"]))
    debug_print("Context !AFTER! the answer: "+str(len(context.user_data["chat_history"]))+"\n"+json.dumps(context.user_data["chat_history"], indent=4))

#!! # Summarize [summarize_depth] answer
#     if len(context.user_data["chat_history"]) >= summarize_depth and len(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']) > 100 :
#         summarized_answer = bart_large_cnn_samsum(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']))
#         debug_print("BEFORE summarizing: "+str(len(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer'])))+"\n"+str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']))
#         await context.bot.send_message(chat_id=update.effective_chat.id, text=str(len(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']))))
#         debug_print("AFTER summarizing:\n"+summarized_answer)
#         context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer'] = summarized_answer
#         debug_print("Context AFTER Summarizing: "+str(len(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer'])))+"\n"+json.dumps(context.user_data["chat_history"], indent=4))
#         await context.bot.send_message(chat_id=update.effective_chat.id, text=str(len(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']))))
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
# String processing defs
def construct_context_string_from_history(chat_context):
# Construct context from history
    context_string = " "
    if len(chat_context) > 0 :
        for i in range(len(chat_context)):
#            context_string = context_string + (chat_context[i]['question']+" "+chat_context[i]['answer'])
            context_string = context_string + ("<s>[INST] "+chat_context[i]['question']+" [/INST] "+chat_context[i]['answer']+" </s>")
    return context_string

def count_words_in_string(string):
    return len(string.split())

# -----------------------------------------------------------------------------------------
# Russian translation
# /tr command - Chat via russian Translator
async def gpt_ru(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_bot_ru_answer = ""
    description = facebook_wmt19_ru_en(" ".join(context.args)).strip()
#    debug_print(description)
    chat_bot_en_answer = Llama_2_13b_chat_hf(description)
#    chat_bot_en_answer = Llama_2_13B_chat_GPTQ(description)
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

# -----------------------------------------------------------------------------------------
# /txt command - Text Generator + Translator
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_bot_ru_answer = "\n_________\nПеревод:\n_________\n"
    chat_bot_en_answer = chat_phi_1_5(" ".join(context.args)).strip()
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
    description = facebook_wmt19_ru_en(" ".join(context.args)).strip()
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

# /init command - sets environment variable INITIAL_PROMPT context from input
async def new_initial_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    os.environ['INITIAL_PROMPT'] = str(" ".join(context.args)).strip()
    await context.bot.send_message(chat_id=update.effective_chat.id, text="New INITIAL_PROMPT:\n"+os.getenv('INITIAL_PROMPT'))
# /reset chat context and print HELP_MESSAGE
async def reset_initial_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["chat_history"] = []
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Chat reset!\n"+HELP_MESSAGE)
# -----------------------------------------------------------------------------------------
# /s Summarize user message with bart_large_cnn_samsum
# -----------------------------------------------------------------------------------------
async def summarize_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = str(" ".join(context.args)).strip()
#    debug_print("user_message to summarize: "+user_message)
    summary = bart_large_cnn_samsum(user_message)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Summary of message:\n"+summary)

# -----------------------------------------------------------------------------------------
# Image classificator and 2x Upscaler
# -----------------------------------------------------------------------------------------
async def photo_classification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    new_file    = await update.message.effective_attachment[-1].get_file()
    file_name   = new_file.file_path.split("/")[-1]
    file_path   = await new_file.download_to_drive(custom_path="data/images/"+file_name)
    description = image_category_32_384(file_path)
    en_text     = "Most likely it's " + description
    ru_text     = facebook_wmt19_en_ru(en_text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=en_text + "\n" + ru_text)
    # Upscaler
    upscaled_picture = stable_diffusion_x4_upscaler(description, str(file_path))
#    debug_print(upscaled_picture)
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=upscaled_picture)

# -----------------------------------------------------------------------------------------
# Audio processing
# -----------------------------------------------------------------------------------------
async def voice_transcribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    new_file    = await update.message.effective_attachment.get_file()
    file_name   = new_file.file_path.split("/")[-1]
    while True:
        try:
            file_path   = await new_file.download_to_drive(custom_path="data/voice/"+file_name)
            break
        except shutil.SameFileError:
            # TODO parse file_name for better making unic name
            file_name = "1" + file_name
    await context.bot.send_message(chat_id=update.effective_chat.id, text=openai_whisper_large_v3(file_path))

# -----------------------------------------------------------------------------------------
# Error message
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.\n" + HELP_MESSAGE)

def debug_print(to_print):
    print("\n!!!------------------------\n"+to_print+"\n------------------------!!!\n")


if __name__ == '__main__':

    # Create a persistence object
    bot_persistence = PicklePersistence(filepath='data/chat/chat_history')

# Variables ------------------------------------------------------------------------------
    HELP_MESSAGE        = os.getenv('HELP_MESSAGE')
    TELEGRAM_BOT_TOKEN  = os.getenv('TELEGRAM_BOT_TOKEN')

    application     = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).persistence(persistence=bot_persistence).build()

    start_handler   = CommandHandler('start', help_command)
    application.add_handler(start_handler)
    help_handler    = CommandHandler('help', help_command)
    application.add_handler(help_handler)

# Image processing -----------------------------------------------------------------------------------------
    img_handler     = CommandHandler('img',     image_generation)
    application.add_handler(img_handler)
    imgh_handler    = CommandHandler('imgh',    image_refine_generation)
    application.add_handler(imgh_handler)
    img_ru_handler  = CommandHandler('imgr',    image_ru_generation)
    application.add_handler(img_ru_handler)
    image_handler   = MessageHandler(filters.PHOTO, photo_classification)
    application.add_handler(image_handler)

# Chat -----------------------------------------------------------------------------------------------------
    tb_llama_2_handler   = CommandHandler('t',    tb_llama_2)
    application.add_handler(tb_llama_2_handler)

    meta_llama_2_handler = CommandHandler('tm',    meta_llama_2)
    application.add_handler(meta_llama_2_handler)



    gpt_ru_handler  = CommandHandler('tr',   gpt_ru)
    application.add_handler(gpt_ru_handler)

    echo_handler    = CommandHandler('txt',  echo)
    application.add_handler(echo_handler)
    echo_ru_handler = CommandHandler('txtr', echo_ru)
    application.add_handler(echo_ru_handler)

    new_context_handler     = CommandHandler('init', new_initial_prompt)
    application.add_handler(new_context_handler)

    reset_context_handler     = CommandHandler('reset', reset_initial_prompt)
    application.add_handler(reset_context_handler)

    summarize_message_handler = CommandHandler('s',summarize_message)
    application.add_handler(summarize_message_handler)

# Audio processing -----------------------------------------------------------------------------------------
    audio_handler   = MessageHandler(filters.VOICE , voice_transcribe)
    application.add_handler(audio_handler)

# Other handlers
    unknown_handler = MessageHandler(filters.COMMAND, error)
    application.add_handler(unknown_handler)

    application.run_polling()
