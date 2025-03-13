
import os
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Path to EmoSet
EMOSET_PATH = "/path/to/emoset"
OUTPUT_PATH = "./joy_samples"

# Load EmoSet annotations
annotations = pd.read_csv(os.path.join(EMOSET_PATH, "annotations.csv"))

# Extract "joy" images
joy_samples = annotations[annotations['label'] == 'joy']

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Copy "joy" samples to output folder
for index, row in tqdm(joy_samples.iterrows(), total=len(joy_samples)):
    src = os.path.join(EMOSET_PATH, "images", row['filename'])
    dst = os.path.join(OUTPUT_PATH, row['filename'])
    shutil.copy(src, dst)

print(f"✅ Extracted {len(joy_samples)} 'joy' samples to {OUTPUT_PATH}")
