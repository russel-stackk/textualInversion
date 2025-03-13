from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model with new embedding
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.load_textual_inversion("./joy-embedding")
pipe = pipe.to("cuda")

# Generate an image using the new token
prompt = "A person smiling with a feeling of <joy-emotion>"

image = pipe(prompt).images[0]

# Display result in Jupyter Notebook
plt.figure(figsize=(5, 5))
plt.imshow(image)
plt.axis("off")
plt.show()
