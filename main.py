import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS configuration to allow your GitHub pages website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    image_url: str

@app.get("/")
def read_root():
    return {"status": "Vidtor Engine Live Node Active"}

@app.post("/v1/generate-video")
async def generate_video(payload: VideoRequest):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="HF_TOKEN Environment Variable Missing!")
    
    hf_url = "https://api-inference.huggingface.co/models/stabilityai/stable-video-diffusion-img2vid-xt"
    
    # Simple setup for authorization
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Preparing input data
            input_data = payload.image_url
            
            # Hit Hugging Face Inference API
            response = await client.post(
                hf_url, 
                headers=headers, 
                json={"inputs": input_data}
            )
            
            if response.status_code == 503:
                raise HTTPException(status_code=503, detail="Model Loading on Hugging Face. Please retry.")
            elif response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"Hugging Face Error: {response.text}")
            
            # Convert bytes content to hex string for easy frontend handling
            video_hex = response.content.hex()
            return {"video_bytes": video_hex}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

