# https://huggingface.co/google/vit-base-patch32-384
# Vision Transformer (base-sized model)

from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
#import requests

def image_category_32_384(image_path):
#    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#    image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open(image_path)
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-384')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch32-384')
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]
