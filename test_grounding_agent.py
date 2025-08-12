# test_grounded_sam.py

import torch
import cv2
import sys
import os
from agents.grounded_sam_agent import query_grounding_dino, query_sam
#from agents.grounded_sam_agent import query_grounded_sam

sys.path.append('./tools/grounded-sam/Grounded-Segment-Anything')
from GroundingDINO.groundingdino.util.inference import load_model,Model
from segment_anything.segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# ----------------------------
# Config & Paths
# ----------------------------
args = {
    "dino": {
        "BOX_THRESHOLD": 0.25,
        "TEXT_THRESHOLD": 0.25
    },
    "sam": {
        "BOX_THRESHOLD": 0.25,
        "TEXT_THRESHOLD": 0.25,
        "NMS_THRESHOLD": 0.8
    }
}

GROUNDING_DINO_CONFIG = "./tools/grounded-sam/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CKPT = "./tools/grounded-sam/Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth"

SAM_ENCODER_VERSION = "vit_h"
SAM_CKPT = "./tools/grounded-sam/Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth"

IMAGE_PATH = "./tools/grounded-sam/Grounded-Segment-Anything/assets/demo2.jpg"
CLASS_PROMPT = ["The running dog"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load models
# ----------------------------
print("[INFO] Loading GroundingDINO...")
grounding_dino_model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CKPT)

print("[INFO] Loading SAM...")
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CKPT)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)
sam_mask_generator = SamAutomaticMaskGenerator(sam)

# ----------------------------
# Test 1: GroundingDINO only
# ----------------------------
print("[TEST] Running GroundingDINO only...")
image_source, boxes, logits, phrases = query_grounding_dino(
    device=DEVICE,
    args=args,
    model=grounding_dino_model,
    image_path=IMAGE_PATH,
    text_prompt=". ".join(CLASS_PROMPT) + ".",
    save_image=True
)
print(f"Detected {len(boxes)} boxes from GroundingDINO")

# ----------------------------
# Test 2: SAM mask generation (full image)
# ----------------------------
print("[TEST] Running SAM mask generation...")
masks = query_sam(
    device=DEVICE,
    args=args,
    sam_mask_generator=sam_mask_generator,
    image=cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
)
print(f"Generated {len(masks)} masks from SAM")


model = Model(model_config_path=GROUNDING_DINO_CONFIG, model_checkpoint_path=GROUNDING_DINO_CKPT)



# import cv2
# from agents.grounded_sam_agent import GroundedSAMAgent

# agent = GroundedSAMAgent(
#     grounding_dino_config="tools/grounded-sam/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
#     grounding_dino_ckpt="tools/grounded-sam/Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth",
#     sam_encoder_version="vit_h",
#     sam_ckpt="tools/grounded-sam/Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth"
# )

# # Detection only
# detections, det_img = agent.detect("demo2.jpg", ["The running dog"])
# cv2.imwrite("output_detect_only.jpg", det_img)

# # Full detect + segment
# detections, seg_img = agent.detect_and_segment("demo2.jpg", ["The running dog"])
# cv2.imwrite("output_detect_segment.jpg", seg_img)
