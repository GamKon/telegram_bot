# https://huggingface.co/TheBloke/Llama-2-13B-Chat-GPTQ

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#from auto_gptq import exllama_set_max_input_length


def Llama_2_13B_chat_GPTQ(user_prompt, context, initial_prompt):
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
    # To use a different branch, change revision
    # For example: revision="main"
#    revision = "main"
#    revision = "gptq-8bit-64g-actorder_True"
#    revision = "gptq-4bit-32g-actorder_True"
#    revision = "gptq-4bit-64g-actorder_True"
#    revision = "gptq-4bit-128g-actorder_True"
    revision = "gptq-8bit-128g-actorder_False"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision=revision)
#    model = exllama_set_max_input_length(model, max_input_length=4096)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# Error
# RuntimeError: The temp_state buffer is too small in the exllama backend for GPTQ with act-order. Please call the exllama_set_max_input_length function to increase the buffer size for a sequence length >=2081:
# from auto_gptq import exllama_set_max_input_length
# model = exllama_set_max_input_length(model, max_input_length=2081)


    # prompt = "Tell me about AI"

    # Old context
    # You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

    # Prompt with context of Interiewer
    # prompt_template=f'''[INST] <<SYS>>
    # You will act as an interviewer. You help people practice interviews, asking them one question at a time and following up if answers aren't clear. Just ask about the position to apply for and you conduct a simple screening interview for the position.
    # I need you to help people practice interviews. They may be interviewing for lots of different jobs, but general question and answer practice will still help them prepare! Make sure to ask questions professionally, as if you were conducting a phone screen for the position the describe. Just ask one question at a time, and be sure to respond to any questions they have and follow up with additional questions relevant to their answers.
    # <</SYS>>
    # {prompt}[/INST]
    # '''

    #prompt_template=f'''[INST] <<SYS>>{initial_prompt} {context}<</SYS>>{user_prompt}[/INST]'''

    full_templated_prompt=f'''[INST] <<SYS>>
    {initial_prompt}
    <</SYS>> [/INST]
    {context}
    <s>[INST] {user_prompt} [/INST]

    '''

    # print("----------------------------------------------prompt to AI-----------------------------------------------------")
    # print(prompt_template)
    # print("---------------------------------------------------------------------------------------------------------------")

    # How many words and tokens are in the full_prompt?
    num_words   = len(full_templated_prompt.split())
    num_tokens  = len(tokenizer.tokenize(full_templated_prompt))


    # # print("\n\n*** Generate:")
    # input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    # output = model.generate(
    #     inputs=input_ids,
    #     temperature=0.7,
    #     do_sample=True,
    #     top_p=0.95,
    #     top_k=40,
    #     max_new_tokens=512
    # )
    # return tokenizer.decode(output[0])
    # print(tokenizer.decode(output[0]))
    # , skip_special_tokens=True


    # Inference can also be done using transformers' pipeline
    # print("*** Pipeline:")
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

    try:
        answer = pipe(full_templated_prompt)[0]['generated_text']
    except RuntimeError as e:
        return f"Plese, repeat the question.\nRuntimeError: {e}", 0, 0
    # print("-------------------------------------------------ANSWER--------------------------------------------------------")
    # print(answer)
    # print("---------------------------------------------------------------------------------------------------------------")
    return answer, num_tokens, num_words
