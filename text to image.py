import os


os.system('pip install diffusers transformers torch accelerate')
import torch
from diffusers import StableDiffusionPipeline


device = "cuda" if torch.cuda.is_available() else "cpu"

# pre-trained Stable Diffusion model from Hugging Face
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)
s

def generate_image(prompt, output_path="generated_image.png"):
    
    if device == "cuda":
        with torch.autocast("cuda"):
            image = pipe(prompt).images[0]
    else:
       
        image = pipe(prompt).images[0]
    
    
    image.save(output_path)
    print(f"Image saved to {output_path}")





prompt = "A futuristic city with flying cars and neon lights"
generate_image(prompt)



