"""
Quick test script for CLIP-Count agent
"""
import cv2
import sys
import os

# Add paths
sys.path.append('.')
from agents.clip_count_agent import ClipCountAgent

def test_clip_count(image_path, object_to_count, ckpt_path):
    """
    Test CLIP-Count agent
    """
    print("="*60)
    print("CLIP-Count Test")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Object to count: {object_to_count}")
    print(f"Checkpoint: {ckpt_path}")
    print()
    
    # Load image
    image_np = cv2.imread(image_path)
    if image_np is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"✓ Image loaded: {image_np.shape[1]}x{image_np.shape[0]} pixels")
    
    # Initialize agent
    print("\nInitializing CLIP-Count agent...")
    agent = ClipCountAgent(ckpt_path=ckpt_path, device="cuda")
    print("✓ Agent initialized")
    
    # Run detection
    print(f"\nCounting '{object_to_count}'...")
    count_result, heatmap = agent.detect_count(image_np, object_to_count)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Count result: {count_result}")
    print(f"Heatmap type: {type(heatmap)}")
    
    # Extract number from result
    if isinstance(count_result, str) and "[estimated count]" in count_result:
        count_number = int(count_result.split("]")[1].strip())
        print(f"Extracted count: {count_number}")
    
    # Save heatmap
    if heatmap is not None:
        output_path = "test_clip_count_output.jpg"
        
        # Convert PIL RGB to numpy BGR for OpenCV
        import numpy as np
        if hasattr(heatmap, 'save'):  # PIL Image
            img_np = np.array(heatmap)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, img_bgr)
            print(f"✓ Heatmap saved to: {output_path}")
    
    print("="*60)

if __name__ == "__main__":
    # Configuration
    image_path = "test_image.jpg"  # Change this
    object_to_count = "fruits"     # Change this
    ckpt_path = os.getenv("CLIP_COUNT_CKPT", "weights/clipcount_pretrained.ckpt")
    
    # Allow command line args
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    if len(sys.argv) >= 3:
        object_to_count = sys.argv[2]
    
    test_clip_count(image_path, object_to_count, ckpt_path)