# https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GPTQ

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def Mistral_7B_OpenOrca_GPTQ(user_prompt, context, initial_prompt):
    model_name_or_path = "TheBloke/Mistral-7B-OpenOrca-GPTQ"
    # To use a different branch, change revision
    # For example: revision="gptq-4bit-32g-actorder_True"
    revision = "gptq-8bit-32g-actorder_True"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision=revision)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    prompt_template=f'''<|im_start|>system
    {initial_prompt}<|im_end|>
    {context}
    <|im_start|>user
    {user_prompt}<|im_end|>
    <|im_start|>assistant
    '''


    print("----------------------------------------------prompt to AI-----------------------------------------------------")
    print(prompt_template)
    print("---------------------------------------------------------------------------------------------------------------")

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
