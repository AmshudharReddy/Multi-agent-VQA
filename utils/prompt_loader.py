import os

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

def load_prompt(prompt_name: str) -> str:
    """
    Loads a prompt template file from the prompts folder.
    Args:
        prompt_name (str): Name without extension, e.g., "parsing", "reattempt"
    Returns:
        str: File contents as a string.
    """
    file_path = os.path.join(PROMPT_DIR, f"{prompt_name}.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()
