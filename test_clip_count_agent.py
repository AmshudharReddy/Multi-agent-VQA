import cv2
from agents.clip_count_agent import ClipCountAgent

# Path to checkpoint
ckpt_path = "tools/clip/CLIP-Count/weights/clipcount_pretrained.ckpt"

# Init agent
agent = ClipCountAgent(ckpt_path, device="cpu")

# Load test image
image_path = "C:\\Users\\amula\\Desktop\\COCO_val2014_000000000357.jpg"
image_np = cv2.imread(image_path)

# Run detection
needed_object = "person"
count, heatmap_overlay = agent.detect_count(image_np, needed_object)

print(f"Detected count: {count}")
heatmap_overlay.save("clipcount_overlay.jpg")
