from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer
import torch
from PIL import Image
from tqdm import tqdm
import os

# Load pre-trained Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Set up textual inversion training
EMBEDDING_NAME = "joy-emotion"
SAVE_PATH = "./joy-embedding"
NUM_STEPS = 500
LEARNING_RATE = 5e-4

# Create a placeholder token
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
placeholder_token = f"<{EMBEDDING_NAME}>"
tokenizer.add_tokens([placeholder_token])
pipe.text_encoder.resize_token_embeddings(len(tokenizer))

# Load images for training
TRAIN_DIR = "./joy_samples"
images = [Image.open(os.path.join(TRAIN_DIR, f)).convert("RGB") for f in os.listdir(TRAIN_DIR)]

# Train the textual inversion embedding
from diffusers.training_utils import TextualInversionTrainer

trainer = TextualInversionTrainer(
    pipe=pipe,
    images=images,
    placeholder_token=placeholder_token,
    learning_rate=LEARNING_RATE,
    num_steps=NUM_STEPS,
    output_dir=SAVE_PATH
)

trainer.train()

# Save the embedding
trainer.save(SAVE_PATH)

print(f"✅ Training complete! Embedding saved to {SAVE_PATH}")
