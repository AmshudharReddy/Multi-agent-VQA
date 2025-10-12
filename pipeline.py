import cv2
import os
import sys
import torch
import re
import warnings
import numpy as np

# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", message=".*autocast.*")

from agents.lvlm_agent import LVLMAgent
from agents.llm_agent import LLMAgent
from agents.clip_count_agent import ClipCountAgent
from agents.grounded_sam_agent import GroundedSAMAgent, query_grounding_dino
from agents.ocr_agent import OCRAgent
# from utils.token_constants import ANSWER_TOKEN, ANSWER_FAILED_TOKEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_pipeline(image_path, question, gemini_api_key, clip_count_ckpt, grounding_dino_config, 
                 grounding_dino_ckpt, sam_ckpt=None, ocr_model_path=None, mode="auto"):
    """
    Multi-Agent VQA Pipeline with direct agent testing support.
    
    Args:
        mode: "auto" (default LVLM-first flow) | "ocr" | "clip_count" | "grounded_sam" | "lvlm"
    """
    
    print(f"[PIPELINE] Running in mode: {mode}")
    
    # =============================
    # DIRECT OCR TESTING
    # =============================
    if mode == "ocr":
        print("[PIPELINE] Direct OCR mode activated")
        ocr_agent = OCRAgent(device="cuda" if torch.cuda.is_available() else "cpu", mode="auto")
        result = ocr_agent.detect_text(image_path)
        
        return {
            "answer": result["extracted_text"],
            "text_blocks": result["text_blocks"],
            "output_image_path": None,  # Could save annotated image if needed
            "agent": "ocr"
        }
    
    # =============================
    # DIRECT CLIP-COUNT TESTING
    # =============================
    elif mode == "clip_count":
        print("[PIPELINE] Direct CLIP-Count mode activated")
        clip_count = ClipCountAgent(ckpt_path=clip_count_ckpt)
        
        # Extract object to count from question (simple heuristic)
        # You can also use LLM here if needed
        llm = LLMAgent(api_key=gemini_api_key)
        object_to_count = llm.get_objects_to_count(question=question)
        print(f"[CLIP-Count] Counting: {object_to_count}")
        
        image_np = cv2.imread(image_path)
        count_result, annotated_img = clip_count.detect_count(image_np, object_to_count)
        
        # ClipCountAgent returns: count_result as string "[estimated count] X" and PIL Image (RGB)
        # Extract the actual count number
        if isinstance(count_result, str) and "[estimated count]" in count_result:
            count_number = int(count_result.split("]")[1].strip())
        else:
            count_number = count_result
        
        # Save annotated image - convert PIL RGB to BGR numpy for saving
        output_path = None
        if annotated_img is not None:
            try:
                # Convert PIL Image (RGB) to numpy array (BGR) for OpenCV
                if hasattr(annotated_img, 'save'):  # It's a PIL Image
                    output_path = "output_clip_count.jpg"
                    # PIL Image is in RGB, convert to BGR for OpenCV
                    img_np = np.array(annotated_img)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, img_bgr)
                    print(f"[CLIP-Count] Saved annotated image to {output_path}")
                else:
                    print(f"[CLIP-Count] Warning: Unexpected annotated image type: {type(annotated_img)}")
            except Exception as e:
                print(f"[CLIP-Count] Warning: Could not save annotated image: {e}")
        
        return {
            "answer": f"Count of {object_to_count}: {count_number}",
            "count": count_number,
            "object": object_to_count,
            "output_image_path": output_path,
            "agent": "clip_count"
        }
    
    # =============================
    # DIRECT GROUNDED-SAM TESTING
    # =============================
    elif mode == "grounded_sam":
        print("[PIPELINE] Direct Grounded-SAM mode activated")
        
        # Initialize Grounded-SAM agent (using your existing agent)
        grounded_sam = GroundedSAMAgent(
            grounding_dino_config=grounding_dino_config,
            grounding_dino_ckpt=grounding_dino_ckpt,
            sam_encoder_version="vit_h",
            sam_ckpt=sam_ckpt,
            device=DEVICE
        )
        
        # Parse classes/objects from question
        # Question can be: "person. car. dog" or "detect people and cars"
        if "." in question:
            # Direct class list format: "person. car. dog"
            classes = [cls.strip() for cls in question.split(".") if cls.strip()]
        else:
            # Natural language - try to extract objects using LLM
            try:
                llm = LLMAgent(api_key=gemini_api_key)
                extracted = llm.extract_needed_objects(question=question, answer="")
                classes = [cls.strip() for cls in extracted.split(".") if cls.strip()]
            except:
                # Fallback: use the whole question as single class
                classes = [question]
        
        print(f"[Grounded-SAM] Classes to detect: {classes}")
        
        # Run detection + segmentation
        try:
            detections, annotated_img = grounded_sam.detect_and_segment(
                image_path=image_path,
                classes=classes,
                box_threshold=0.25,
                text_threshold=0.25,
                nms_threshold=0.8
            )
            
            # Save annotated image
            output_path = "output_grounded_sam.jpg"
            cv2.imwrite(output_path, annotated_img)
            print(f"[Grounded-SAM] Saved annotated image to {output_path}")
            
            # Count detections per class
            class_counts = {}
            for class_id in detections.class_id:
                class_name = classes[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Build answer
            if len(detections) == 0:
                answer = f"No objects matching '{', '.join(classes)}' were detected."
            else:
                count_str = ", ".join([f"{count} {cls}" for cls, count in class_counts.items()])
                answer = f"Detected {len(detections)} objects: {count_str}"
            
            print(f"[Grounded-SAM] {answer}")
            
            return {
                "answer": answer,
                "detected_objects": [classes[cid] for cid in detections.class_id],
                "class_counts": class_counts,
                "num_detections": len(detections),
                "output_image_path": output_path,
                "agent": "grounded_sam"
            }
            
        except Exception as e:
            print(f"[Grounded-SAM] Error during detection: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error during object detection: {str(e)}",
                "detected_objects": [],
                "output_image_path": None,
                "agent": "grounded_sam"
            }
    
    # =============================
    # DIRECT LVLM TESTING
    # =============================
    elif mode == "lvlm":
        print("[PIPELINE] Direct LVLM mode activated")
        lvlm = LVLMAgent(api_key=gemini_api_key)
        direct_answer = lvlm.ask_directly(image_path, question)
        
        return {
            "answer": direct_answer,
            "agent": "lvlm"
        }
    
    # =============================
    # AUTO MODE (Original Pipeline)
    # =============================
    else:
        print("[PIPELINE] Auto mode - running full pipeline with LVLM first")
        return run_auto_pipeline(
            image_path, question, gemini_api_key, clip_count_ckpt, 
            grounding_dino_config, grounding_dino_ckpt, sam_ckpt, ocr_model_path
        )


def run_auto_pipeline(image_path, question, gemini_api_key, clip_count_ckpt, 
                      grounding_dino_config, grounding_dino_ckpt, sam_ckpt=None, ocr_model_path=None):
    """
    Original pipeline logic - LVLM first, then fallback to specialized agents
    """
    print("[PIPELINE] Initializing agents...")
    # Init Agents
    lvlm = LVLMAgent(api_key=gemini_api_key)
    llm = LLMAgent(api_key=gemini_api_key)
    clip_count = ClipCountAgent(ckpt_path=clip_count_ckpt)

    # Init Grounded-SAM agent (using your existing agent)
    grounded_sam = GroundedSAMAgent(
        grounding_dino_config=grounding_dino_config,
        grounding_dino_ckpt=grounding_dino_ckpt,
        sam_encoder_version="vit_h" if sam_ckpt else None,
        sam_ckpt=sam_ckpt,
        device=DEVICE
    )

    # Step 1: Ask directly
    print("[PIPELINE] Step 1: LVLM asking directly...")
    direct_answer = lvlm.ask_directly(image_path, question)
    print(f"[LVLM Direct Answer] {direct_answer}")

    is_answer_failed = re.search(r'\[Answer Failed\]', direct_answer) is not None or re.search(r'\[Numeric Answer Failed\]', direct_answer) is not None
    is_counting_problem = re.search(r'\[Numeric Answer\]', direct_answer) is not None or re.search(r'\[Numeric Answer Failed\]', direct_answer) is not None

    if is_answer_failed:
        print("[PIPELINE] LVLM reported failure to answer.")

        if is_counting_problem:
            print("[PIPELINE] Detected counting problem. Extracting object to count...")
            object_to_count = llm.get_objects_to_count(question=question)
            print(f"[LLM Extracted Object to Count] {object_to_count}")

            print("[PIPELINE] Running CLIP-Count agent...")
            image_np = cv2.imread(image_path)
            reattempted_answer, _ = clip_count.detect_count(image_np, object_to_count)
            
            # Extract count number from "[estimated count] X" format
            if isinstance(reattempted_answer, str) and "[estimated count]" in reattempted_answer:
                count_number = int(reattempted_answer.split("]")[1].strip())
                reattempted_answer = f"There are {count_number} {object_to_count} in the image."
            
            print(f"[CLIP-Count Result] {reattempted_answer}")

            return {"answer": reattempted_answer, "agent": "clip_count"}

        else:
            print("[PIPELINE] Non-counting problem. Extracting needed objects...")
            needed_objects = llm.extract_needed_objects(question=question, answer=direct_answer)
            print(f"[LLM Extracted Needed Objects] {needed_objects}")

            # Parse classes from needed_objects
            classes = [cls.strip() for cls in needed_objects.split(".") if cls.strip()]
            print(f"[PIPELINE] Classes to detect: {classes}")

            print("[PIPELINE] Running Grounded-SAM for object detection...")
            try:
                detections, annotated_img = grounded_sam.detect_and_segment(
                    image_path=image_path,
                    classes=classes,
                    box_threshold=0.25,
                    text_threshold=0.25,
                    nms_threshold=0.8
                )
                
                print(f"[Grounded-SAM] Detected {len(detections)} objects.")
                
                # Build phrases for LVLM
                phrases = [classes[cid] for cid in detections.class_id]
                boxes = detections.xyxy  # Already in absolute coordinates
                
                print("[PIPELINE] Generating object descriptions with LVLM...")
                object_attributes = lvlm.object_description(
                    image_path=image_path,
                    bboxes=boxes,
                    phrases=phrases,
                    question=question
                )
                print(f"[LVLM Object Descriptions] {object_attributes}")

                print("[PIPELINE] Reattempting answer with LVLM...")
                reattempted_answer = lvlm.reattempt(
                    image_path=image_path,
                    question=question,
                    prev_answer=direct_answer,
                    obj_descriptions=object_attributes
                )
                print(f"[LVLM Reattempted Answer] {reattempted_answer}")

                return {"answer": reattempted_answer, "agent": "grounded_sam+lvlm"}
                
            except Exception as e:
                print(f"[PIPELINE] Error in Grounded-SAM: {e}")
                import traceback
                traceback.print_exc()
                return {"answer": f"Error during object detection: {str(e)}", "agent": "error"}

    else:
        print("[PIPELINE] LVLM answered successfully on first attempt.")
        return {"answer": direct_answer, "agent": "lvlm"}