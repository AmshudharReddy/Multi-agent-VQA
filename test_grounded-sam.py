"""
Test script for Grounded-SAM agent (using your existing agent)
"""
import os
import sys
import cv2
sys.path.append('.')

from agents.grounded_sam_agent import GroundedSAMAgent

def test_grounded_sam(image_path, classes):
    """
    Test Grounded-SAM detection and segmentation
    
    Args:
        image_path: Path to test image
        classes: List of class names to detect (e.g., ["person", "car", "dog"])
    """
    print("="*60)
    print("Grounded-SAM Test")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Classes: {classes}")
    print()
    
    # Configuration from environment or defaults
    grounding_dino_config = os.getenv(
        "GROUNDING_DINO_CONFIG",
        "tools/grounded-sam/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    grounding_dino_ckpt = os.getenv(
        "GROUNDING_DINO_CKPT",
        "tools/grounded-sam/Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth"
    )
    sam_ckpt = os.getenv(
        "SAM_CKPT",
        "tools/grounded-sam/Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth"
    )
    
    print(f"Config: {grounding_dino_config}")
    print(f"Grounded DINO checkpoint: {grounding_dino_ckpt}")
    print(f"SAM checkpoint: {sam_ckpt}")
    print()
    
    # Check if files exist
    if not os.path.exists(grounding_dino_config):
        print(f"❌ Config not found: {grounding_dino_config}")
        return
    if not os.path.exists(grounding_dino_ckpt):
        print(f"❌ Grounded DINO checkpoint not found: {grounding_dino_ckpt}")
        return
    if not os.path.exists(sam_ckpt):
        print(f"❌ SAM checkpoint not found: {sam_ckpt}")
        return
    
    # Initialize agent
    print("Initializing Grounded-SAM agent...")
    try:
        agent = GroundedSAMAgent(
            grounding_dino_config=grounding_dino_config,
            grounding_dino_ckpt=grounding_dino_ckpt,
            sam_encoder_version="vit_h",
            sam_ckpt=sam_ckpt,
            device="cuda"  # Change to "cpu" if no GPU
        )
        print("✓ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 1: Detection only
    print("\n" + "-"*60)
    print("TEST 1: Detection Only")
    print("-"*60)
    try:
        detections, annotated_img = agent.detect(
            image_path=image_path,
            classes=classes,
            box_threshold=0.25,
            text_threshold=0.25,
            nms_threshold=0.8
        )
        
        print(f"✓ Detected {len(detections)} objects")
        
        # Save detection result
        output_path = "test_detection_only.jpg"
        cv2.imwrite(output_path, annotated_img)
        print(f"✓ Saved detection image to: {output_path}")
        
        # Show detections per class
        class_counts = {}
        for class_id in detections.class_id:
            class_name = classes[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\nDetected objects:")
        for cls, count in class_counts.items():
            print(f"  - {cls}: {count}")
            
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Detection + Segmentation
    print("\n" + "-"*60)
    print("TEST 2: Detection + Segmentation")
    print("-"*60)
    try:
        detections, annotated_img = agent.detect_and_segment(
            image_path=image_path,
            classes=classes,
            box_threshold=0.25,
            text_threshold=0.25,
            nms_threshold=0.8
        )
        
        print(f"✓ Detected and segmented {len(detections)} objects")
        
        # Save segmentation result
        output_path = "test_detection_segmentation.jpg"
        cv2.imwrite(output_path, annotated_img)
        print(f"✓ Saved segmentation image to: {output_path}")
        
        # Show detections per class
        class_counts = {}
        for class_id in detections.class_id:
            class_name = classes[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\nSegmented objects:")
        for cls, count in class_counts.items():
            print(f"  - {cls}: {count}")
        
        print(f"\nBounding boxes shape: {detections.xyxy.shape}")
        print(f"Masks shape: {detections.mask.shape}")
            
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

if __name__ == "__main__":
    # Default test
    image_path = "test_image.jpg"
    classes = ["person", "car"]
    
    # Parse command line arguments
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        # Classes can be passed as: "person car dog" or "person. car. dog"
        classes_str = " ".join(sys.argv[2:])
        if "." in classes_str:
            classes = [c.strip() for c in classes_str.split(".") if c.strip()]
        else:
            classes = classes_str.split()
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("\nUsage:")
        print("  python test_grounded_sam.py <image_path> <class1> <class2> ...")
        print("\nExamples:")
        print('  python test_grounded_sam.py image.jpg person car')
        print('  python test_grounded_sam.py image.jpg "person. car. dog"')
        sys.exit(1)
    
    test_grounded_sam(image_path, classes)