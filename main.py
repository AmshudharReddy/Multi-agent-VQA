import os
from dotenv import load_dotenv
from pipeline import run_pipeline

# Load environment variables
load_dotenv()

# ===== USER INPUTS =====
IMAGE_PATH = "C:/Users/pra21/Desktop/101 apples/T/107926.jpg"
QUESTION = "What the man in the back is doing?"

# Read from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLIP_COUNT_CKPT = os.getenv("CLIP_COUNT_CKPT")
GROUNDING_DINO_CONFIG = os.getenv("GROUNDING_DINO_CONFIG")
GROUNDING_DINO_CKPT = os.getenv("GROUNDING_DINO_CKPT")
SAM_CKPT = os.getenv("SAM_CKPT")

# ===== RUN PIPELINE =====
result = run_pipeline(
    image_path=IMAGE_PATH,
    question=QUESTION,
    gemini_api_key=GEMINI_API_KEY,
    clip_count_ckpt=CLIP_COUNT_CKPT,
    grounding_dino_config=GROUNDING_DINO_CONFIG,
    grounding_dino_ckpt=GROUNDING_DINO_CKPT,
    sam_ckpt=SAM_CKPT
)

print("\n[FINAL RESULT]")
print(result)
