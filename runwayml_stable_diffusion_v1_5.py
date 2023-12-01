from diffusers import StableDiffusionPipeline
import torch

def stable_diffusion_v1_5(prompt):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    #pipe = pipe.to("cpu")

    # = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]

    image_name = prompt.replace(" ", "_")+".png"
    image.save(image_name)
#    print(image_name)
    return image_name
