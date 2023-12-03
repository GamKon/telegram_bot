# Gaivoronsky/ruGPT-3.5-13B-fp16

from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('Gaivoronsky/ruGPT-3.5-13B-fp16')
tokenizer = AutoTokenizer.from_pretrained('Gaivoronsky/ruGPT-3.5-13B-fp16')
#model = model.half()
model = model.to('cuda:0')

request = "Человек: Сколько весит жираф? Помощник: "
encoded_input = tokenizer(request, return_tensors='pt', \
                          add_special_tokens=False).to('cuda:0')
output = model.generate(
    **encoded_input,
    num_beams=2,
    do_sample=True,
    max_new_tokens=100
)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# export PATH=$PATH:/home/gamkon/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/
# export LD_LIBRARY_PATH=/home/gamkon/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/

#CUDA exception! Error code: no CUDA-capable device is detected
