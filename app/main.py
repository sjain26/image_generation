# app/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.models import GenerationRequest
from app.utils import encode_image, decode_image
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import logging
from typing import Optional
import os
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Generation API",
    description="API for AI-powered image generation and style transfer",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()  # Reduce memory usage
    logger.info(f"Model loaded successfully on {DEVICE}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Style prompts dictionary
STYLE_PROMPTS = {
    "anime": "anime style, highly detailed, vibrant colors, Studio Ghibli inspired",
    "oil_painting": "oil painting style, detailed brushstrokes, artistic, renaissance style",
    "watercolor": "watercolor painting, soft colors, flowing textures, artistic",
    "digital_art": "digital art style, clean lines, modern, professional",
    "sketch": "pencil sketch, detailed linework, artistic drawing",
    "default": ""
}

def cleanup_memory():
    """Helper function to clean up memory"""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

@app.get("/")
async def read_root():
    """Root endpoint to verify API is running"""
    return {
        "status": "online",
        "model": MODEL_ID,
        "device": DEVICE,
        "available_styles": list(STYLE_PROMPTS.keys())
    }

@app.post("/generate")
async def generate_image(request: GenerationRequest):
    """Generate an image based on the provided prompt and style"""
    try:
        logger.info(f"Generating image for prompt: {request.prompt}")
        
        # Combine prompt with style
        style_prompt = STYLE_PROMPTS.get(request.style, "")
        modified_prompt = f"{request.prompt}, {style_prompt}".strip()
        
        # Generate image
        with torch.inference_mode():
            image = pipe(
                modified_prompt,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale
            ).images[0]
        
        # Encode image to base64
        image_base64 = encode_image(image)
        
        # Clean up memory
        cleanup_memory()
        
        return {
            "status": "success",
            "image": image_base64,
            "metadata": {
                "prompt": modified_prompt,
                "steps": request.steps,
                "guidance_scale": request.guidance_scale
            }
        }
    
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        cleanup_memory()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/style-transfer")
async def style_transfer(
    file: UploadFile = File(...),
    style: str = "default",
    strength: float = 0.75
):
    """Apply style transfer to an uploaded image"""
    try:
        logger.info(f"Applying style transfer with style: {style}")
        
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG and PNG images are supported"
            )
        
        # Read and validate image
        image_data = await file.read()
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )
        
        # Implement style transfer logic here
        # For now, we'll just return the original image
        # In a production environment, you would implement proper
        # style transfer using models like CycleGAN
        
        # Encode result
        result_base64 = encode_image(image)
        
        # Clean up memory
        cleanup_memory()
        
        return {
            "status": "success",
            "image": result_base64,
            "metadata": {
                "style": style,
                "strength": strength
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in style transfer: {str(e)}")
        cleanup_memory()
        raise HTTPException(status_code=500, detail=str(e))

# Test endpoints
@app.get("/test/health")
async def health_check():
    """Check if the service is healthy"""
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "device": DEVICE
    }

@app.get("/test/styles")
async def get_styles():
    """Get available style options"""
    return {
        "styles": list(STYLE_PROMPTS.keys()),
        "descriptions": STYLE_PROMPTS
    }

# tests/test_main.py remains the same