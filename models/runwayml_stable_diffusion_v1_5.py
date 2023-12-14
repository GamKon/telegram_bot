# https://huggingface.co/runwayml/stable-diffusion-v1-5
# text-to-image diffusion model

from diffusers import StableDiffusionPipeline
import torch

def stable_diffusion_v1_5(prompt, file_path, num_inference_steps):
    model_id    = "runwayml/stable-diffusion-v1-5"
    pipe        = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe        = pipe.to("cuda")
    # pipe        = pipe.to("cpu")

    # prompt      = "a photo of an astronaut riding a horse on mars"
    image       = pipe(prompt).images[0]

    image_name  = prompt.replace(" ", "_")+".png"
    full_path   = file_path+"/"+image_name
    image.save(full_path)
#    print(image_name)
    return full_path
