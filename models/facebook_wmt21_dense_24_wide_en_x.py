# https://huggingface.co/facebook/wmt21-dense-24-wide-en-x

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def facebook_wmt21_en_x(en_ru_input):
    device = "cuda:0" # if torch.cuda.is_available() else "cpu"
    model_id = "facebook/wmt21-dense-24-wide-en-x"
#    model_id = "Helsinki-NLP/opus-mt-en-ru"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True)

    #, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
#, load_in_8bit=True)
#, load_in_4bit=True)
# importlib.metadata.PackageNotFoundError: No package metadata was found for bitsandbytes


#    model.to(device)
# Can't use .to if use load_in_xbit=True option is used in model

    tokenizer = AutoTokenizer.from_pretrained(model_id)#, padding_side="left")
#    tokenizer.pad_token = tokenizer.eos_token

# Put tokenizer on GPU
# .input_ids.to(device)
    inputs = tokenizer(en_ru_input, padding=True, return_tensors="pt").input_ids.to(device)

    # translate English to Russian
    generated_tokens = model.generate(inputs, forced_bos_token_id=tokenizer.get_lang_id("ru"))#, max_new_tokens=1024)
    translated_string = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    # to get string instead of list: translate_string[0]
    return translated_string
