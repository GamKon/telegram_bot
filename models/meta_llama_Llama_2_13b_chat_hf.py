# https://huggingface.co/meta-llama/Llama-2-13b-chat-hf

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def Llama_2_13b_chat_hf(user_prompt, context, initial_prompt):
    model_name_or_path = "meta-llama/Llama-2-13b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main", token=os.getenv("HF_TOKEN"))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, token=os.getenv("HF_TOKEN"))

    #prompt = "Tell me about AI"

    # Template
    # <s>[INST]
    #{{ user_message_1 }} [/INST] {{ llama_answer_1 }} </s><s>[INST] {{ user_message_2 }} [/INST]

    prompt_template=f'''[INST] <<SYS>>
    {initial_prompt}
    <</SYS>> [/INST]
    {context}
    <s>[INST] {user_prompt} [/INST]

    '''

    print("----------------------------------------------prompt to AI-----------------------------------------------------")
    print(prompt_template)
    print("---------------------------------------------------------------------------------------------------------------")

    # print("\n\n*** Generate:")

    # input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    # output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    # print(tokenizer.decode(output[0]))

    # Inference can also be done using transformers' pipeline

    #print("*** Pipeline:")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1
    )

    answer = pipe(prompt_template)[0]['generated_text']
    print("-------------------------------------------------ANSWER--------------------------------------------------------")
    print(answer)
    print("---------------------------------------------------------------------------------------------------------------")
    return answer
