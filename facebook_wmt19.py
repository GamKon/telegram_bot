from transformers import FSMTForConditionalGeneration, FSMTTokenizer

def facebook_wmt19_en_ru(en_ru_input):
    mname = "facebook/wmt19-en-ru"
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)

#    en_ru_input = "Machine learning is great, isn't it?"
    input_ids = tokenizer.encode(en_ru_input, return_tensors="pt")
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#    print(decoded) # Машинное обучение - это здорово, не так ли?
    return decoded

def facebook_wmt19_ru_en(ru_en_input):
    mname = "facebook/wmt19-ru-en"
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)

#    ru_en_input = "Машинное обучение - это здорово, не так ли?"
    input_ids = tokenizer.encode(ru_en_input, return_tensors="pt")
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#    print(decoded) # Machine learning is great, isn't it?
    return decoded

#print(facebook_wmt19_en_ru("Machine learning is great, isn't it?"))
#print(facebook_wmt19_ru_en("Машинное обучение - это здорово, не так ли?"))
