from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import tempfile
import base64
from typing import Optional
from pipeline import run_pipeline  # Your existing pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Multi-Agent VQA API", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (your frontend HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLIP_COUNT_CKPT = os.getenv("CLIP_COUNT_CKPT")
GROUNDING_DINO_CONFIG = os.getenv("GROUNDING_DINO_CONFIG")
GROUNDING_DINO_CKPT = os.getenv("GROUNDING_DINO_CKPT")
SAM_CKPT = os.getenv("SAM_CKPT")

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML page"""
    return FileResponse("static/index.html")

@app.post("/api/vqa")
async def vqa_endpoint(
    question: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Process Visual Question Answering request
    
    Args:
        question: The question to ask about the image
        image: The uploaded image file
        
    Returns:
        JSON response with answer and optional output image
    """
    
    # Validate inputs
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Create temporary file for the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        # Read and save the uploaded image
        contents = await image.read()
        temp_file.write(contents)
        temp_image_path = temp_file.name
    
    try:
        # Run your pipeline
        result = run_pipeline(
            image_path=temp_image_path,
            question=question,
            gemini_api_key=GEMINI_API_KEY,
            clip_count_ckpt=CLIP_COUNT_CKPT,
            grounding_dino_config=GROUNDING_DINO_CONFIG,
            grounding_dino_ckpt=GROUNDING_DINO_CKPT,
            sam_ckpt=SAM_CKPT
        )
        
        # Check if result contains output image path
        output_image_b64 = None
        if isinstance(result, dict) and 'output_image_path' in result:
            try:
                with open(result['output_image_path'], 'rb') as img_file:
                    output_image_b64 = base64.b64encode(img_file.read()).decode('utf-8')
            except Exception as e:
                print(f"Error encoding output image: {e}")
        
        return {
            "answer": str(result),
            "output_image": output_image_b64,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_image_path)
        except Exception as e:
            print(f"Error cleaning up temp file: {e}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Multi-Agent VQA API is running"}

# Optional: Add endpoint to get model status
@app.get("/api/models/status")
async def model_status():
    """Check if all required models and API keys are available"""
    status = {
        "gemini_api_key": bool(GEMINI_API_KEY),
        "clip_count_ckpt": bool(CLIP_COUNT_CKPT and os.path.exists(CLIP_COUNT_CKPT)),
        "grounding_dino_config": bool(GROUNDING_DINO_CONFIG and os.path.exists(GROUNDING_DINO_CONFIG)),
        "grounding_dino_ckpt": bool(GROUNDING_DINO_CKPT and os.path.exists(GROUNDING_DINO_CKPT)),
        "sam_ckpt": bool(SAM_CKPT and os.path.exists(SAM_CKPT))
    }
    
    all_ready = all(status.values())
    
    return {
        "all_models_ready": all_ready,
        "model_status": status
    }

if __name__ == "__main__":
    import uvicorn
    import warnings
    
    # Suppress the timm deprecation warning
    warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
    
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    print("Starting Multi-Agent VQA Server...")
    print("Frontend available at: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "file:app",  # Use import string instead of app object
        host="0.0.0.0", 
        port=8000,
        reload=True  # This will now work properly
    )