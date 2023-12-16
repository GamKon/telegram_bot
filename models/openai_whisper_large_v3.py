import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
#from datasets import load_dataset

def openai_whisper_large_v3(audio_file):
    device = "cuda:0"# if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    #dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    #sample = dataset[0]["audio"]
    #result = pipe(sample)

# Just transcription
    #result = pipe(str(audio_file)

    result = pipe(str(audio_file), generate_kwargs={"task": "translate"})

    #print(result["text"])
    return result["text"]

# Whisper predicts the language of the source audio automatically.
# If the source audio language is known a-priori, it can be passed as an argument to the pipeline:
    # result = pipe(sample, generate_kwargs={"language": "english"})

# By default, Whisper performs the task of speech transcription,
# where the source audio language is the same as the target text language.
# To perform speech translation, where the target text is in English, set the task to "translate":
    # result = pipe(sample, generate_kwargs={"task": "translate"})
