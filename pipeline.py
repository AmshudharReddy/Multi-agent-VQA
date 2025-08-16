import cv2
import os
import sys
import torch
import re

from agents.lvlm_agent import LVLMAgent
from agents.llm_agent import LLMAgent
from agents.clip_count_agent import ClipCountAgent
from agents.grounded_sam_agent import GroundedSAMAgent, query_grounding_dino
# from utils.token_constants import ANSWER_TOKEN, ANSWER_FAILED_TOKEN

sys.path.append('./tools/grounded-sam/Grounded-Segment-Anything')
from GroundingDINO.groundingdino.util.inference import load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_pipeline(image_path, question, gemini_api_key, clip_count_ckpt, grounding_dino_config, grounding_dino_ckpt, sam_ckpt=None):

    print("[PIPELINE] Initializing agents...")
    # Init Agents
    lvlm = LVLMAgent(api_key=gemini_api_key)
    llm = LLMAgent(api_key=gemini_api_key)
    clip_count = ClipCountAgent(ckpt_path=clip_count_ckpt)

    #load grounding dino model
    print("[PIPELINE] Loading GroundingDINO model...")
    grounding_dino = load_model(grounding_dino_config, grounding_dino_ckpt)

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
            print(f"[CLIP-Count Result] {reattempted_answer}")

            return reattempted_answer

        else:
            print("[PIPELINE] Non-counting problem. Extracting needed objects...")
            needed_objects = llm.extract_needed_objects(question=question, answer=direct_answer)
            print(f"[LLM Extracted Needed Objects] {needed_objects}")

            print("[PIPELINE] Running GroundingDINO for object detection...")
            image, boxes, logits, phrases = query_grounding_dino(
                device=DEVICE,
                args=args,
                model=grounding_dino,
                image_path=image_path,
                text_prompt=needed_objects
            )
            print(f"[GroundingDINO] Detected {len(boxes)} objects.")
            print(f"[GroundingDINO Phrases] {phrases}")

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

            return reattempted_answer

    else:
        print("[PIPELINE] LVLM answered successfully on first attempt.")
        return direct_answer
