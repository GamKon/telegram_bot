# https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
# text-to-image diffusion model

# To use the whole base + refiner pipeline as an ensemble of experts you can run:

from diffusers import DiffusionPipeline
import torch

# base + refiner pipeline
def stable_diffusion_xl_base_refiner_1_0(prompt, file_path):
    # load both base & refiner
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    base.to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")

    # Define how many steps and what % of steps to be run on each experts (80/20) here
    # n_steps = 40
    # high_noise_frac = 0.8
#    prompt = "A majestic lion jumping from a big stone at night"

    # ! Crashes here:
    # base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
    # refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

    base_n_steps =20 #(20)
    base_high_noise_frac = 0.8
    refiner_n_steps = 20 #(6)
    refiner_high_noise_frac = 0.8
    # run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=base_n_steps,
        denoising_end=base_high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=refiner_n_steps,
        denoising_start=refiner_high_noise_frac,
        image=image,
    ).images[0]

    image_name  = prompt.replace(" ", "_")+".png"
    full_path   = file_path+"/"+image_name
    image.save(full_path)
#    print(image_name)
    return full_path


# To just use the base model, you can run:
# from diffusers import DiffusionPipeline
# import torch
def stable_diffusion_xl_base_1_0(prompt, file_path):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16")
    pipe.to("cuda")

    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    # prompt = "An astronaut riding a green horse"

    #high_noise_frac = 0.8
    n_steps = 90

    image = pipe(
        prompt=prompt,
        num_inference_steps=n_steps,
#        denoising_end=high_noise_frac,
    ).images[0]

    image_name  = prompt.replace(" ", "_")+".png"
    full_path   = file_path+"/"+image_name
    image.save(full_path)
#    print(image_name)
    return full_path
