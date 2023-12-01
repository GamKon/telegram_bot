# Disable tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

def image_category(image_path):

    image = Image.open(image_path)

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


def main():
    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)
#    image_path = 'images/kaggle_image_dataset/_train/flower/flower_0688.jpg'
#    image_path = 'images/kaggle_image_dataset/_train/motorbike/motorbike_0742.jpg'
    image_path = 'images/kaggle_image_dataset/_train/car/car_0614.jpg'
    image = Image.open(image_path)

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

if __name__ == '__main__':
    main()
