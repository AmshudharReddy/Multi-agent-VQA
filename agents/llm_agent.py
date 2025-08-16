import os
import google.generativeai as genai
from utils.prompt_loader import load_prompt

class LLMAgent:
    def __init__(self, api_key=None, model_name="gemini-2.0-flash"):
        """
        api_key: Google Generative AI API key
        model_name: Gemini text-only model
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key required for LLM Agent (Google Generative AI)")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _send_request(self, prompt, max_tokens=500):
        """Helper: send text prompt to Gemini and return raw response."""
        response = self.model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens}
        )
        return response.text.strip()

    
    def check_if_number_is_large(self,answer):
        prompt_template = load_prompt("message_to_check_if_the_number_is_large")
        prompt = prompt_template.format(answer=answer)
        return self._send_request(prompt)
    
    def get_objects_to_count(self,question):
        prompt_template = load_prompt("message_to_get_objects_for_counting")
        prompt = prompt_template.format(question=question)
        return self._send_request(prompt)
    
    def extract_needed_objects(self, question, answer):
        prompt_template = load_prompt("messages_to_extract_needed_objects")
        prompt = prompt_template.format(question=question, answer=answer)
        return self._send_request(prompt)
