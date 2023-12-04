from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
import torch
from PIL import Image
#from io import BytesIO

def sd_x2_latent_upscaler(prompt, file_path):
    # pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    # pipeline.to("cuda")

    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "stabilityai/sd-x2-latent-upscaler",
        torch_dtype=torch.float16)
    upscaler.to("cuda")

    #prompt = "a photo of an astronaut high resolution, unreal engine, ultra realistic"
    generator = torch.manual_seed(33)

    # we stay in latent space! Let's make sure that Stable Diffusion returns the image
    # in latent space
    #low_res_latents = pipeline(prompt, generator=generator, output_type="latent").images

    image = Image.open(file_path)#.convert("RGB")

    upscaled_image = upscaler(
        prompt="big lion on the beach",
        image=image,
        num_inference_steps=20,
        guidance_scale=0,
        generator=generator,
    ).images[0]

    #image = pipe(prompt=prompt).images[0]

    #image_name  = prompt.replace(" ", "_")+".png"
    #full_path   = file_path+"/"+image_name
    upscaled_image.save(file_path)
#    print(image_name)
    return file_path

    # Let's save the upscaled image under "upscaled_astronaut.png"
    # upscaled_image.save("astronaut_1024.png")

    # # as a comparison: Let's also save the low-res image
    # with torch.no_grad():
    #     image = pipeline.decode_latents(low_res_latents)
    # image = pipeline.numpy_to_pil(image)[0]

    # image.save("astronaut_512.png")
