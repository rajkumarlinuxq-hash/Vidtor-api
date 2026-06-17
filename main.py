import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Vidtor AI Cloud Core Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-video-diffusion-img2vid-xt"

class VideoPayload(BaseModel):
    image_url: str

@app.get("/")
def health_check():
    return {"status": "Vidtor AI Neural Network is Online", "engine": "SVD-XT"}

@app.post("/v1/generate-video")
async def generate_video(payload: VideoPayload):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Cloud Secret Token is missing!")
    
    try:
        img_response = requests.get(payload.image_url)
        if img_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
        
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        ai_response = requests.post(API_URL, headers=headers, data=img_response.content)
        
        if ai_response.status_code != 200:
            raise HTTPException(status_code=ai_response.status_code, detail="AI Node busy or overloaded")
            
        return {"video_bytes": ai_response.content.hex(), "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
