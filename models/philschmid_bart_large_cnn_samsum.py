# https://huggingface.co/philschmid/bart-large-cnn-samsum
# philschmid/bart-large-cnn-samsum

from transformers import pipeline

def bart_large_cnn_samsum(text_to_summarize):

    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

    max_new_tokens = len(text_to_summarize.split())
    if max_new_tokens < 300:
        return text_to_summarize
    else:
        return summarizer(
            text_to_summarize,
            min_new_tokens = 300,
            max_new_tokens = max_new_tokens
    #        max_length=max_length
        )[0]['summary_text']


# conversation = '''Jeff: Can I train a 🤗 Transformers model on Amazon SageMaker?
#     Philipp: Sure you can use the new Hugging Face Deep Learning Container.
#     Jeff: ok.
#     Jeff: and how can I get started?
#     Jeff: where can I find documentation?
#     Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face
#     '''

#print(summarizer(text_to_summarize)[0]['summary_text'])
