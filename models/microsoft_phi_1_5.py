# https://huggingface.co/microsoft/phi-1_5
# Text generator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def chat_phi_1_5(chat_message):
    torch.set_default_device("cuda")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
    inputs = tokenizer(chat_message, return_tensors="pt", return_attention_mask=False)

    outputs = model.generate(**inputs, max_length=550)
    text = tokenizer.batch_decode(outputs)[0]
    return text
#    print(text)

# If you need to use the model in a lower precision (e.g., FP16),
# please wrap the model's forward pass with torch.autocast(), as follows:

#        with torch.autocast(model.device.type, dtype=torch.float16, enabled=True):
#            outputs = model.generate(**inputs, max_length=200)
