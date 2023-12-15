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
from models.facebook_wmt21_dense_24_wide_en_x import facebook_wmt21_en_x
from models.runwayml_stable_diffusion_v1_5 import stable_diffusion_v1_5
from models.openai_whisper_large_v3 import openai_whisper_large_v3
from models.stabilityai_stable_diffusion_xl_base_1_0 import stable_diffusion_xl_base_1_0, stable_diffusion_xl_base_refiner_1_0
from models.stabilityai_stable_diffusion_x4_upscaler import stable_diffusion_x4_upscaler
from models.TheBloke_Llama_2_13B_Chat_GPTQ import Llama_2_13B_chat_GPTQ
from models.meta_llama_Llama_2_13b_chat_hf import Llama_2_13b_chat_hf
from models.philschmid_bart_large_cnn_samsum import bart_large_cnn_samsum
from models.TheBloke_Mistral_7B_OpenOrca_GPTQ import Mistral_7B_OpenOrca_GPTQ

# -----------------------------------------------------------------------------------------
# Command handlers
# -----------------------------------------------------------------------------------------
# /start and /help commans
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Print HELP_MESSAGE
    await context.bot.send_message(chat_id=update.effective_chat.id, text=HELP_MESSAGE)
# -----------------------------------------------------------------------------------------
# /tm command - send question to Mistral-7B-GPTQ q8
async def mistral_7b(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Get question, stripping double spaces
    user_prompt = str(" ".join(context.args)).strip()
    await bot_ask_mistral(user_prompt, context, update)
# -----------------------------------------------------------------------------------------
# /t command - send question to Llama_2-13B-chat-GPTQ q4
async def tb_llama_2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Get question, stripping double spaces
    user_prompt = str(" ".join(context.args)).strip()
    await bot_ask_llama_2(user_prompt, context, update)
# -----------------------------------------------------------------------------------------
# /init command - sets user's system initial_prompt from input or environment variable INITIAL_PROMPT
async def new_initial_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "initial_prompt" not in context.user_data.keys():
        context.user_data["initial_prompt"] = []
        context.user_data["initial_prompt"].append({"system_prompt": ""})
    if len(context.args) == 0:
        context.user_data["initial_prompt"][0] = {"system_prompt": str(os.getenv('INITIAL_PROMPT')).strip()}
    else:
        context.user_data["initial_prompt"][0] = {"system_prompt": str(" ".join(context.args)).strip()}
    await context.bot.send_message(chat_id=update.effective_chat.id, text="New initial_prompt:\n"+context.user_data["initial_prompt"][0]["system_prompt"])
# -----------------------------------------------------------------------------------------
# /reset chat history
async def reset_chat_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["chat_history"] = []
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Chat reset!")
# -----------------------------------------------------------------------------------------
# /drop last question - answer from history
async def drop_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        context.user_data["chat_history"].pop()
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Last dialog dropped!")
    except IndexError:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="History is empty!")
        pass
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# Chat
# -----------------------------------------------------------------------------------------
async def bot_ask_mistral(user_prompt, context, update):
    if "chat_history" not in context.user_data.keys():
        context.user_data["chat_history"] = []
    if "initial_prompt" not in context.user_data.keys():
        context.user_data["initial_prompt"] = []
        context.user_data["initial_prompt"].append({"system_prompt": str(os.getenv('INITIAL_PROMPT')).strip()})

    # Construct context from history
    context_string_long = construct_context_string_from_history(context.user_data["chat_history"], "Mistral")
# Drop old messages from history if whole string for a model is longer than 2048 words
###    context_string = await drop_old_messages_from_history(user_prompt, context_string_long, 2048, context, update, "Mistral")
    context_string = context_string_long

    # 10 tries to handle crashes. Most likely, because out of RAM
    for i in range(1, 10):
        try:
            # Ask Mistral
            gpt_answer, num_tokens, num_words = Mistral_7B_OpenOrca_GPTQ(user_prompt, context_string, context.user_data["initial_prompt"][0]["system_prompt"])
            break
        except (ValueError, RuntimeError) as e:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Mistral crash,\n" + str(e) + "\nRetry "+str(i))
            if i == 9:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="FATAL Mistral crash after "+str(i)+" retries\nSuggest to /reset chat history")
                return
            else: sleep(5)

    # Get the answer from the end of the returned context
    gpt_answer = str(gpt_answer.split("assistant")[-1])
    # Sent answer to the chat
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Index: "+str(len(context.user_data["chat_history"]))+"; Words: "+str(num_words)+"; Tokens: "+str(num_tokens)+"\n"+gpt_answer)
    #await telegram_output_long_string(gpt_answer, context, update)
    # Translate the answer
# TRANSLATION DISABLED
#    gpt_answer_ru = await translate_string_en_ru(gpt_answer)
    # Sent answer to the chat
#    await context.bot.send_message(chat_id=update.effective_chat.id, text="\n-----------------\n "+gpt_answer_ru)
    #await telegram_output_long_string(gpt_answer_ru, context, update)


    # Save question and answer to list of dictionaries
    context.user_data["chat_history"].append({"question": emoji.replace_emoji(user_prompt, replace=''), "answer": emoji.replace_emoji(gpt_answer.strip(), replace='')})

##    debug_print("Context !AFTER! the answer: "+str(len(context.user_data["chat_history"]))+"\n"+json.dumps(context.user_data["chat_history"], indent=4))

# How deep to summarize
    summarize_depth = 8
# Summarize [summarize_depth] answer
    await summarize_answer(update, context, summarize_depth)

# -----------------------------------------------------------------------------------------
async def bot_ask_llama_2(user_prompt, context, update):
    if "chat_history" not in context.user_data.keys():
        context.user_data["chat_history"] = []
    if "initial_prompt" not in context.user_data.keys():
        context.user_data["initial_prompt"] = []
        context.user_data["initial_prompt"].append({"system_prompt": str(os.getenv('INITIAL_PROMPT')).strip()})

    # Construct context from history
    context_string_long = construct_context_string_from_history(context.user_data["chat_history"], "Llama")
# Drop old messages from history if whole string for a model is longer than 1020 words
#    context_string = await drop_old_messages_from_history(user_prompt, context_string_long, 2048, context, update, "Llama")
    context_string = context_string_long
        # Llama 2 documentation: To increase temp_state buffer:
        # from auto_gptq import exllama_set_max_input_length
        # model = exllama_set_max_input_length(model, max_input_length=xxxx)

    # 10 tries to handle crashes. Most likely, because out of RAM
    for i in range(1, 10):
        try:
            # Ask Llama_2
            gpt_answer, num_tokens, num_words = Llama_2_13B_chat_GPTQ(user_prompt, context_string, context.user_data["initial_prompt"][0]["system_prompt"])
            # If model crashed, drop the answer with error description
            if "Plese, repeat the question.\nRuntimeError:" in gpt_answer:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=gpt_answer)
                return
            break
        except (ValueError, RuntimeError) as e:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Llama_2 crash,\n" + str(e) + "\nRetry "+str(i))
            if i == 9:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="FATAL Llama_2 crash after "+str(i)+" retries\nSuggest to /reset chat history")
                return
            else: sleep(5)

    # Get the answer from the end of the returned context
    gpt_answer = str(gpt_answer.split("[/INST]")[-1])

    # If Telegram truncates long message message
        # Split output by 4096 symbols
        # answers = [gpt_answer[i:i + 4096] for i in range(len(gpt_answer), 4096)]
        # for answer in answers:
        #     await context.bot.send_message(chat_id=update.effective_chat.id, text=str(len(context.user_data["chat_history"]))+" "+answer)

    # Sent answer to the chat
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Index: "+str(len(context.user_data["chat_history"]))+"; Words: "+str(num_words)+"; Tokens: "+str(num_tokens)+"\n"+gpt_answer)
    # Save question and answer to list of dictionaries
    context.user_data["chat_history"].append({"question": emoji.replace_emoji(user_prompt, replace=''), "answer": emoji.replace_emoji(gpt_answer.strip(), replace='')})

    debug_print("Context !AFTER! the answer: "+str(len(context.user_data["chat_history"]))+"\n"+json.dumps(context.user_data["chat_history"], indent=4))

    # How deep to summarize
    summarize_depth = 6
# Summarize [summarize_depth] answer
#    await summarize_answer(update, context, summarize_depth)

# -----------------------------------------------------------------------------------------
# String processing defs
# -----------------------------------------------------------------------------------------
def construct_context_string_from_history(chat_context, model_format):
# Construct context from history
    context_string = " "
    if len(chat_context) > 0 :
        for i in range(len(chat_context)):
            if model_format == "Llama":
                context_string = context_string + ("<s>[INST] "+chat_context[i]['question']+" [/INST] "+chat_context[i]['answer']+" </s>")
            elif model_format == "Mistral":
#            context_string = context_string + (chat_context[i]['question']+" "+chat_context[i]['answer'])
                context_string = context_string + ("<|im_start|>user "+chat_context[i]['question']+" <|im_end|> <|im_start|>assistant "+chat_context[i]['answer']+" <|im_end|> ")
            else:
                debug_print("ERROR: Unknown model format")
                context_string = " "
    return context_string

def count_words_in_string(string):
    return len(string.split())

# -----------------------------------------------------------------------------------------
async def drop_old_messages_from_history(user_prompt, context_string, max_words_in_context, context, update, model_format):
    # Drop old messages from history if whole string for a model is longer than {max_words_in_context} words
    while len(str(user_prompt+context_string+context.user_data["initial_prompt"][0]["system_prompt"]).split())+24 > max_words_in_context:
        context.user_data["chat_history"].pop(0)
        debug_message = str(len(str(user_prompt+context_string+context.user_data["initial_prompt"][0]["system_prompt"]).split())+24) + " > " + str(max_words_in_context) + "\nPOP oldest record"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=debug_message)
        context_string = construct_context_string_from_history(context.user_data["chat_history"], model_format)
    words_in_string = "Words sent to model - "+str(len(str(user_prompt+context_string+context.user_data["initial_prompt"][0]["system_prompt"]).split())+24)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=words_in_string)
    return context_string

# -----------------------------------------------------------------------------------------
async def summarize_answer(update, context, summarize_depth):
    if len(context.user_data["chat_history"]) >= summarize_depth and len(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']) > 300 :
        summarized_answer = bart_large_cnn_samsum(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']))
#        debug_print("BEFORE summarizing: "+str(len(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer'])))+"\n"+str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']))
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Summarizing answer #"+str(len(context.user_data["chat_history"])-summarize_depth)+"\nLenght: "+str(len(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']))))
#        debug_print("AFTER summarizing:\n"+summarized_answer)
        context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer'] = summarized_answer
#        debug_print("Context AFTER Summarizing: "+str(len(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer'])))+"\n"+json.dumps(context.user_data["chat_history"], indent=4))
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Result length: "+str(len(str(context.user_data["chat_history"][len(context.user_data["chat_history"])-summarize_depth]['answer']))))

# -----------------------------------------------------------------------------------------
# /s Summarize user message with bart_large_cnn_samsum
# -----------------------------------------------------------------------------------------
async def summarize_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = str(" ".join(context.args)).strip()
#    debug_print("user_message to summarize: "+user_message)
    summary = bart_large_cnn_samsum(user_message)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Summary of message:\n"+summary)

# -----------------------------------------------------------------------------------------
async def telegram_output_long_string(string_to_post, context, update):
    # If Telegram truncates long message message
    # Split output by 4096 symbols
    answers = [string_to_post[i:i + 4096] for i in range(len(string_to_post), 4096)]
    for answer in answers:
        print(answer)
        #await context.bot.send_message(chat_id=update.effective_chat.id, text=str(len(context.user_data["chat_history"]))+" "+answer)

# -----------------------------------------------------------------------------------------
# Translate string from English to Russian
async def translate_string_en_ru(string_en):
    string_ru = ""
    # Translate by paragraphs
    paragraphs = string_en.split("\n")
    for paragraph in paragraphs:
        if len(paragraph) > 2 :
            translation = facebook_wmt21_en_x(paragraph)
            string_ru = string_ru + " ".join(translation) + "\n\n"
    return string_ru


# -----------------------------------------------------------------------------------------
# May not working commands. Didn't check in a while
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
    chat_bot_ru_answer = chat_bot_ru_answer + facebook_wmt21_en_x(chat_bot_en_answer)
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
        chat_bot_ru_answer = chat_bot_ru_answer + facebook_wmt21_en_x(answer_en)
    chat_bot_answer = chat_bot_en_answer + chat_bot_ru_answer
    # Split output by 4096 symbols
    answers = [chat_bot_answer[i:i + 4096] for i in range(0, len(chat_bot_answer), 4096)]
    for answer in answers:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

# -----------------------------------------------------------------------------------------
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
        chat_bot_ru_answer = chat_bot_ru_answer + facebook_wmt21_en_x(answer_en)
    # Split output by 4096 symbols
    answers = [chat_bot_ru_answer[i:i + 4096] for i in range(0, len(chat_bot_ru_answer), 4096)]
    for answer in answers:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


# -----------------------------------------------------------------------------------------
# Image generators
# -----------------------------------------------------------------------------------------
# /img command - Image generator
# stable_diffusion_xl_base_1_0
async def image_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # If first argument is a number, use it as num_inference_steps
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

    generated_picture = stable_diffusion_v1_5(description, "data/generated_images", num_inference_steps)
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=generated_picture, caption=description_ru+"\n"+description_ru)


# -----------------------------------------------------------------------------------------
# Image classificator and 4x Upscaler
# -----------------------------------------------------------------------------------------
async def photo_classification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    new_file    = await update.message.effective_attachment[-1].get_file()
    file_name   = new_file.file_path.split("/")[-1]
    file_path   = await new_file.download_to_drive(custom_path="data/images/"+file_name)
    description = image_category_32_384(file_path)
    en_text     = "Most likely it's " + description
    ru_text     = facebook_wmt21_en_x(en_text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=en_text + "\n" + ru_text)
    # Upscaler
    upscaled_picture = stable_diffusion_x4_upscaler(description, str(file_path))
#    debug_print(upscaled_picture)
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=upscaled_picture)


# -----------------------------------------------------------------------------------------
# Audio processing
# -----------------------------------------------------------------------------------------
# Transcript voice message with Oopenai Whisper-large-v3
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
    user_prompt = openai_whisper_large_v3(file_path)
    if "hey" in str(user_prompt.split()[0]).lower() and "bot" in str(user_prompt.split()[1]).lower():
        # cut first two words in string user_prompt and send to bot
        user_prompt = " ".join(user_prompt.split()[2:])
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Asking Mistral: "+user_prompt)
        await bot_ask_mistral(user_prompt, context, update)
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=user_prompt)


# -----------------------------------------------------------------------------------------
# Error message
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.\n" + HELP_MESSAGE)
# -----------------------------------------------------------------------------------------
# Debug print string
def debug_print(to_print):
    print("\n!!!------------------------\n"+to_print+"\n------------------------!!!\n")


# -----------------------------------------------------------------------------------------
#           MAIN
# -----------------------------------------------------------------------------------------
if __name__ == '__main__':

    # Create a persistence object
    bot_persistence = PicklePersistence(filepath='data/chat/chat_history')

# Variables
    HELP_MESSAGE        = os.getenv('HELP_MESSAGE')
    TELEGRAM_BOT_TOKEN  = os.getenv('TELEGRAM_BOT_TOKEN')

    application     = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).persistence(persistence=bot_persistence).build()

# LLM
    # /t command - send question to Llama_2-13B-chat-GPTQ q4
    tb_llama_2_handler   = CommandHandler('tm',    tb_llama_2)
    application.add_handler(tb_llama_2_handler)

    # /tm command - send question to Mistral-7B-GPTQ q8
    mistral_7b_handler = CommandHandler('t',    mistral_7b)
    application.add_handler(mistral_7b_handler)

    # /tr command - Chat via russian Translator
    gpt_ru_handler  = CommandHandler('tr',   gpt_ru)
    application.add_handler(gpt_ru_handler)

    # /txt command - Text Generator + Translator
    echo_handler    = CommandHandler('txt',  echo)
    application.add_handler(echo_handler)

    # /txtr command - russian only Text Generator + Translator
    echo_ru_handler = CommandHandler('txtr', echo_ru)
    application.add_handler(echo_ru_handler)

    # /init command - sets environment variable INITIAL_PROMPT from input
    new_context_handler     = CommandHandler('init',    new_initial_prompt)
    application.add_handler(new_context_handler)

    # /reset chat history
    reset_context_handler     = CommandHandler('reset', reset_chat_history)
    application.add_handler(reset_context_handler)

    # /drop last question - answer from history
    drop_prompt_handler     = CommandHandler('drop',    drop_prompt)
    application.add_handler(drop_prompt_handler)

    # /s Summarize user message with bart_large_cnn_samsum
    summarize_message_handler = CommandHandler('s',     summarize_message)
    application.add_handler(summarize_message_handler)

# Image processing
    # /img command - Image generator stable_diffusion_xl_base_1_0
    img_handler     = CommandHandler('img',     image_generation)
    application.add_handler(img_handler)

    # /imgh command - Image generator with Refiner stable_diffusion_xl_base_refiner_1_0
    imgh_handler    = CommandHandler('imgh',    image_refine_generation)
    application.add_handler(imgh_handler)

    # /imgr command - Russian description Image Generator
    img_ru_handler  = CommandHandler('imgr',    image_ru_generation)
    application.add_handler(img_ru_handler)

    # Image classificator and 4x Upscaler
    image_handler   = MessageHandler(filters.PHOTO,   photo_classification)
    application.add_handler(image_handler)

# Audio processing
    # Transcript voice message with Oopenai Whisper-large-v3
    audio_handler   = MessageHandler(filters.VOICE ,  voice_transcribe)
    application.add_handler(audio_handler)

# System commands
    # /start and /help commans
    start_handler   = CommandHandler('start', help_command)
    application.add_handler(start_handler)
    help_handler    = CommandHandler('help', help_command)
    application.add_handler(help_handler)

    # Error message
    unknown_handler = MessageHandler(filters.COMMAND, error)
    application.add_handler(unknown_handler)

    application.run_polling()
