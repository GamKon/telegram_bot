# https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler

# import requests
from PIL import Image
# from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

def stable_diffusion_x4_upscaler(prompt, file_path):
    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")

    # let's download an  image
    # url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
    # response = requests.get(url)
    # low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = Image.open(file_path)
    # low_res_img.thumbnail()
    # low_res_img = low_res_img.resize((1280, 720))

    # prompt = "big lion on the beach"

    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
    full_path_new = file_path+"_upscaled.jpg"
    upscaled_image.save(full_path_new)
    return full_path_new
