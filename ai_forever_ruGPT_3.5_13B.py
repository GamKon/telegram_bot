# https://huggingface.co/ai-forever/ruGPT-3.5-13B

import torch
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained('ai-forever/ruGPT-3.5-13B')
tokenizer = AutoTokenizer.from_pretrained('ai-forever/ruGPT-3.5-13B')
model = model.half()
model = model.to('cuda:0')

request = "Гагарин полетел в космос в"

encoded_input = tokenizer(request, return_tensors='pt', \
                          add_special_tokens=False).to('cuda:0')
output = model.generate(
    **encoded_input,
    num_beams=2,
    do_sample=True,
    max_new_tokens=20
)

print(tokenizer.decode(output[0], skip_special_tokens=True))

#-------------------------

# request = "Нейронная сеть — это"

# encoded_input = tokenizer(request, return_tensors='pt', \
#                           add_special_tokens=False).to('cuda:0')
# output = model.generate(
#     **encoded_input,
#     num_beams=4,
#     do_sample=True,
#     max_new_tokens=100
# )

# print(tokenizer.decode(output[0], skip_special_tokens=True))

#-------------------------

# request = "Стих про программиста может быть таким:"

# encoded_input = tokenizer(request, return_tensors='pt', \
#                           add_special_tokens=False).to('cuda')
# output = model.generate(
#     **encoded_input,
#     num_beams=2,
#     do_sample=True,
#     max_new_tokens=100
# )

# print(tokenizer.decode(output[0], skip_special_tokens=True))

#-------------------------

# request = "Человек: Сколько весит жираф? Помощник: "
# encoded_input = tokenizer(request, return_tensors='pt', \
#                           add_special_tokens=False).to('cuda')
# output = model.generate(
#     **encoded_input,
#     num_beams=2,
#     do_sample=True,
#     max_new_tokens=100
# )
# print(tokenizer.decode(output[0], skip_special_tokens=True))
