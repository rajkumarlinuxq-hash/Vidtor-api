import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Frontend ke liye raasta clear (CORS Policy configuration)
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
    
    # Stability AI SVD-XT API Endpoint URL
    hf_url = "https://api-inference.huggingface.co/models/stabilityai/stable-video-diffusion-img2vid-xt"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Hugging Face ko call lagayi
            response = await client.post(
                hf_url, 
                headers=headers, 
                json={"inputs": payload.image_url},
                headers={"Content-Type": "application/json"} if not payload.image_url.startswith("data:") else None
            )
            
            if response.status_code == 503:
                raise HTTPException(status_code=503, detail="Model Loading on Hugging Face. Please retry.")
            elif response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Hugging Face Node Error")
            
            # Binary content ko Hex format string me badla frontend compatibility ke liye
            video_hex = response.content.hex()
            return {"video_bytes": video_hex}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

