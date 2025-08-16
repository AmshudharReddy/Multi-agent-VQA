import os
import cv2
import numpy as np
from torchvision.ops import box_convert
import google.generativeai as genai
from utils.prompt_loader import load_prompt


class LVLMAgent:
    def __init__(self, api_key=None, model_name="gemini-2.0-flash"):
        """
        api_key: Google Generative AI API key
        model_name: Gemini Vision model name
        """
        self.min_bbox_size = 32
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key required for LVLM Agent (Google Generative AI)")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def process_image(self, image, bbox=None):
        # we have to crop the image before converting it to base64
        image = cv2.imread(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if bbox is not None:
            width, height = bbox[2], bbox[3]
            xyxy = box_convert(boxes=bbox, in_fmt="cxcywh", out_fmt="xyxy")
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # increase the receptive field of each box to include possible nearby objects and contexts
            if width < self.min_bbox_size:
                x1 = int(max(0, x1 - (self.min_bbox_size - width) / 2))
                x2 = int(min(image.shape[1], x2 + (self.min_bbox_size - width) / 2))
            if height < self.min_bbox_size:
                y1 = int(max(0, y1 - (self.min_bbox_size - height) / 2))
                y2 = int(min(image.shape[0], y2 + (self.min_bbox_size - height) / 2))

            # cv2.imwrite('test_images/original_image' + str(bbox) + '.jpg', image)
            image = image[y1:y2, x1:x2]
            # cv2.imwrite('test_images/cropped_image' + str(bbox) + '.jpg', image)

        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = np.array(buffer).tobytes()

        return image_bytes

    def _send_request(self, image_path, prompt, bbox=None, max_tokens=300):
        """Helper: send image + prompt to Gemini and return raw response."""
        image_bytes = self.process_image(image_path,bbox)

        response = self.model.generate_content(
            [
                {"mime_type": "image/png", "data": image_bytes},
                prompt
            ],
            generation_config={"max_output_tokens": max_tokens}
        )

        return response.text.strip()

    # 1. Ask directly
    def ask_directly(self, image_path, question):
        prompt_template = load_prompt("ask_directly")
        prompt = prompt_template.format(question=question)
        return self._send_request(image_path, prompt, max_tokens=500)
    
    # 2. describing the objects detected by grounded-sam
    def object_description(self, image_path, bboxes, phrases, question):
        responses = []
        total_num_objects = len(bboxes)
        for i in range(total_num_objects):
            bbox = bboxes[i]
            phrase = phrases[i]
            prompt_template = load_prompt("messages_to_query_object_attributes")
            prompt = prompt_template.format(phrase=phrase,question=question)
            response = self._send_request(image_path, prompt, bbox, max_tokens=400)
            responses.append(response)
        return responses
    
    # 3. reattempting the question along with prev_answer and obj_descriptions
    def reattempt(self, image_path, question, prev_answer, obj_descriptions):
        prompt_template = load_prompt("reattempt")
        prompt = prompt_template.format(question=question,prev_answer=prev_answer)
        for i, obj in enumerate(obj_descriptions):
            prompt += "[Object "+ str(i) + "] "+obj + "; "
        return self._send_request(image_path,prompt,bbox=None, max_tokens=700)
        
        
    
    
