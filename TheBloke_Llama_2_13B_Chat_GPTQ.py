# https://huggingface.co/TheBloke/Llama-2-13B-Chat-GPTQ

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def Llama_2_13B_chat_GPTQ(prompt):
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
    # To use a different branch, change revision
    # For example: revision="main"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # prompt = "Tell me about AI"
    prompt_template=f'''[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>
    {prompt}[/INST]

    '''

    # print("\n\n*** Generate:")

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(
        inputs=input_ids,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        top_k=40,
        max_new_tokens=512
    )

    return tokenizer.decode(output[0] , skip_special_tokens=True)
    # print("\n!!---1---!!\n"+tokenizer.decode(output[0])+"\n!!---1---!!\n")
    # , skip_special_tokens=True


    # Inference can also be done using transformers' pipeline
    # print("*** Pipeline:")
    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=512,
    #     do_sample=True,
    #     temperature=0.7,
    #     top_p=0.95,
    #     top_k=40,
    #     repetition_penalty=1.1
    # )
    # print("\n!!---1---!!\n"+pipe(prompt_template)[0]['generated_text']+"\n!!---1---!!\n")

# Llama_2_13B_chat_GPTQ("What is 3 + 2 ?")
# Llama_2_13B_chat_GPTQ("What did I ask the last time?")
