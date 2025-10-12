from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import tempfile
import base64
from typing import Optional
from pipeline import run_pipeline  # Import your main pipeline
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# -------------------------------
# FASTAPI SETUP
# -------------------------------
app = FastAPI(title="Multi-Agent VQA API", version="1.0.0")

# Enable CORS (frontend can connect freely)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files (like index.html, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------
# ENVIRONMENT VARIABLES
# -------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLIP_COUNT_CKPT = os.getenv("CLIP_COUNT_CKPT")
GROUNDING_DINO_CONFIG = os.getenv("GROUNDING_DINO_CONFIG")
GROUNDING_DINO_CKPT = os.getenv("GROUNDING_DINO_CKPT")
SAM_CKPT = os.getenv("SAM_CKPT")
OCR_MODEL_PATH = os.getenv("OCR_MODEL_PATH", "models/ocr_model.pth")  # Optional custom OCR model

# -------------------------------
# FRONTEND ROUTE
# -------------------------------
@app.get("/")
async def serve_frontend():
    """Serve frontend (optional UI if present in /static)"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "Multi-Agent VQA API is running. Visit /docs for API testing."}

# -------------------------------
# MAIN VQA ENDPOINT
# -------------------------------
@app.post("/api/vqa")
async def vqa_endpoint(
    question: str = Form(...),
    image: UploadFile = File(...),
    mode: Optional[str] = Form("auto")  # 'auto', 'lvlm', 'clip_count', 'grounded_sam', 'ocr'
):
    """
    Multi-Agent Visual Question Answering endpoint with direct agent testing.

    Args:
        question (str): User's query about the image/PDF.
        image (UploadFile): Uploaded image or PDF.
        mode (str): Agent mode
            - 'auto': LVLM first, then fallback to specialized agents (default)
            - 'lvlm': Direct LVLM only
            - 'clip_count': Direct CLIP-Count agent (for counting)
            - 'grounded_sam': Direct Grounded-SAM (for object detection)
            - 'ocr': Direct OCR agent (for text extraction)

    Returns:
        JSON with 'answer', 'agent', and optional 'output_image' (base64-encoded).
    """

    # ---------------------------
    # Validate inputs
    # ---------------------------
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not (image.content_type.startswith('image/') or image.content_type == 'application/pdf'):
        raise HTTPException(status_code=400, detail="File must be an image or PDF")

    # Validate mode
    valid_modes = ['auto', 'lvlm', 'clip_count', 'grounded_sam', 'ocr']
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}"
        )

    # Save temporary uploaded file
    file_ext = os.path.splitext(image.filename)[-1] if image.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        contents = await image.read()
        temp_file.write(contents)
        temp_image_path = temp_file.name

    # ---------------------------
    # Run the pipeline
    # ---------------------------
    try:
        result = run_pipeline(
            image_path=temp_image_path,
            question=question,
            gemini_api_key=GEMINI_API_KEY,
            clip_count_ckpt=CLIP_COUNT_CKPT,
            grounding_dino_config=GROUNDING_DINO_CONFIG,
            grounding_dino_ckpt=GROUNDING_DINO_CKPT,
            sam_ckpt=SAM_CKPT,
            ocr_model_path=OCR_MODEL_PATH,
            mode=mode
        )

        # Convert any output image to base64 for web preview
        output_image_b64 = None
        if isinstance(result, dict) and 'output_image_path' in result and result['output_image_path']:
            try:
                with open(result['output_image_path'], 'rb') as img_file:
                    output_image_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                # Clean up output image
                os.unlink(result['output_image_path'])
            except Exception as e:
                print(f"[Warning] Could not encode output image: {e}")

        # Build response
        response = {
            "status": "success",
            "mode": mode,
            "agent": result.get("agent", mode) if isinstance(result, dict) else mode,
            "answer": str(result.get("answer", result) if isinstance(result, dict) else result),
            "output_image": output_image_b64
        }

        # Add extra metadata for specific agents
        if isinstance(result, dict):
            if mode == "clip_count":
                response["count"] = result.get("count")
                response["object"] = result.get("object")
            elif mode == "grounded_sam":
                response["detected_objects"] = result.get("detected_objects")
                response["object_descriptions"] = result.get("object_descriptions")
            elif mode == "ocr":
                response["text_blocks"] = result.get("text_blocks")
                response["detection_method"] = result.get("method")
                response["num_regions"] = len(result.get("text_blocks", []))

        return response

    except Exception as e:
        import traceback
        print(f"[ERROR] Pipeline error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    finally:
        try:
            os.unlink(temp_image_path)
        except Exception as e:
            print(f"[Cleanup] Failed to delete temp file: {e}")

# -------------------------------
# HEALTH CHECK ENDPOINT
# -------------------------------
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Multi-Agent VQA API is running fine!"}

# -------------------------------
# MODEL STATUS ENDPOINT
# -------------------------------
@app.get("/api/models/status")
async def model_status():
    """Check if all required models and API keys are available"""
    status = {
        "gemini_api_key": bool(GEMINI_API_KEY),
        "clip_count_ckpt": bool(CLIP_COUNT_CKPT and os.path.exists(CLIP_COUNT_CKPT)),
        "grounding_dino_config": bool(GROUNDING_DINO_CONFIG and os.path.exists(GROUNDING_DINO_CONFIG)),
        "grounding_dino_ckpt": bool(GROUNDING_DINO_CKPT and os.path.exists(GROUNDING_DINO_CKPT)),
        "sam_ckpt": bool(SAM_CKPT and os.path.exists(SAM_CKPT)),
        "ocr_model": bool(OCR_MODEL_PATH and os.path.exists(OCR_MODEL_PATH))
    }

    all_ready = all(status.values())
    return {"all_models_ready": all_ready, "model_status": status}

# -------------------------------
# AVAILABLE MODES ENDPOINT
# -------------------------------
@app.get("/api/modes")
async def get_available_modes():
    """Get list of available testing modes"""
    return {
        "modes": [
            {
                "name": "auto",
                "description": "LVLM first, then fallback to specialized agents (default pipeline)"
            },
            {
                "name": "lvlm",
                "description": "Direct LVLM testing only"
            },
            {
                "name": "clip_count",
                "description": "Direct CLIP-Count agent for counting objects"
            },
            {
                "name": "grounded_sam",
                "description": "Direct Grounded-SAM for object detection and segmentation"
            },
            {
                "name": "ocr",
                "description": "Direct OCR agent for text extraction"
            }
        ]
    }

# -------------------------------
# MAIN EXECUTION (LOCALHOST)
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    import warnings
    warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")

    os.makedirs("static", exist_ok=True)

    print("🚀 Starting Multi-Agent VQA Server...")
    print("🌐 Frontend available at: http://localhost:8000")
    print("📘 API Docs available at: http://localhost:8000/docs")
    print("\n📋 Available testing modes:")
    print("   - auto: Full pipeline (LVLM first)")
    print("   - lvlm: Direct LVLM testing")
    print("   - clip_count: Direct counting")
    print("   - grounded_sam: Direct object detection")
    print("   - ocr: Direct text extraction")

    uvicorn.run(
        "file:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )