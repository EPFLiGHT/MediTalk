from fastapi import FastAPI, Request, Header, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, Response
import requests
import logging
import os
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MediTalk Web UI", version="1.0.0")

# Service URLs - support both Docker (container names) and local deployment (localhost)
# Docker mode: services are referenced by container names (meditron, orpheus, whisper)
# Local mode: services are on localhost with different ports
MEDITRON_URL = os.getenv("MEDITRON_URL", "http://localhost:5006")
MULTIMEDITRON_URL = os.getenv("MULTIMEDITRON_URL", "http://localhost:5009")
ORPHEUS_URL = os.getenv("ORPHEUS_URL", "http://localhost:5005")
BARK_URL = os.getenv("BARK_URL", "http://localhost:5008")
WHISPER_URL = os.getenv("WHISPER_URL", "http://localhost:5007")

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

# Proxy endpoints for the frontend
@app.api_route("/api/meditron/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_meditron(request: Request, path: str):
    """Proxy requests to Meditron service"""
    logger.info(f"Proxying request to Meditron: {path}")
    try:
        # Get request data
        body = await request.body() if request.method in ["POST", "PUT"] else None
        logger.info(f"Request method: {request.method}, Body length: {len(body) if body else 0}")
        
        # Forward to Meditron service
        response = requests.request(
            method=request.method,
            url=f"{MEDITRON_URL}/{path}",
            headers=dict(request.headers),
            data=body,
            timeout=660  # Increased to 11 minutes (600s + buffer for Meditron processing)
        )

        logger.info(f"Response status: {response.status_code}, Body length: {len(response.content) if response.content else 0}")
        
        return response.json()
    except Exception as e:
        logger.error(f"Error proxying to Meditron: {e}")
        return {"error": str(e)}

@app.api_route("/api/multimeditron/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_multimeditron(request: Request, path: str):
    """Proxy requests to MultiMeditron service"""
    logger.info(f"Proxying request to MultiMeditron: {path}")
    try:
        # Get request data
        body = await request.body() if request.method in ["POST", "PUT"] else None
        logger.info(f"Request method: {request.method}, Body length: {len(body) if body else 0}")
        
        # Forward to MultiMeditron service
        response = requests.request(
            method=request.method,
            url=f"{MULTIMEDITRON_URL}/{path}",
            headers=dict(request.headers),
            data=body,
            timeout=660  # Increased to 11 minutes for multimodal processing
        )

        logger.info(f"Response status: {response.status_code}, Body length: {len(response.content) if response.content else 0}")
        
        return response.json()
    except Exception as e:
        logger.error(f"Error proxying to MultiMeditron: {e}")
        return {"error": str(e)}

@app.api_route("/api/orpheus/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_orpheus(request: Request, path: str):
    """Proxy requests to Orpheus service"""
    logger.info(f"Proxying request to Orpheus: {path}")
    try:
        # Get request data
        body = await request.body() if request.method in ["POST", "PUT"] else None
        logger.info(f"Request method: {request.method}, Body length: {len(body) if body else 0}")
        
        # Forward to Orpheus service
        response = requests.request(
            method=request.method,
            url=f"{ORPHEUS_URL}/{path}",
            headers=dict(request.headers),
            data=body,
            timeout=60
        )
        
        return response.json()
    except Exception as e:
        logger.error(f"Error proxying to Orpheus: {e}")
        return {"error": str(e)}

@app.api_route("/api/bark/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_bark(request: Request, path: str):
    """Proxy requests to Bark TTS service"""
    logger.info(f"Proxying request to Bark: {path}")
    try:
        # Get request data
        body = await request.body() if request.method in ["POST", "PUT"] else None
        logger.info(f"Request method: {request.method}, Body length: {len(body) if body else 0}")
        
        # Forward to Bark service
        response = requests.request(
            method=request.method,
            url=f"{BARK_URL}/{path}",
            headers=dict(request.headers),
            data=body,
            timeout=60
        )
        
        return response.json()
    except Exception as e:
        logger.error(f"Error proxying to Bark: {e}")
        return {"error": str(e)}

@app.api_route("/api/whisper/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_whisper(request: Request, path: str):
    """Proxy requests to Whisper service"""
    logger.info(f"Proxying request to Whisper: {path}")
    try:
        # Get request data
        body = await request.body() if request.method in ["POST", "PUT"] else None
        logger.info(f"Request method: {request.method}, Body length: {len(body) if body else 0}")
        
        # Forward to Whisper service
        response = requests.request(
            method=request.method,
            url=f"{WHISPER_URL}/{path}",
            headers=dict(request.headers),
            data=body,
            timeout=120  # 2 minutes for speech recognition
        )
        
        return response.json()
    except Exception as e:
        logger.error(f"Error proxying to Whisper: {e}")
        return {"error": str(e)}

@app.post("/api/whisper/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Upload audio file to Whisper service for transcription"""
    logger.info(f"Transcribing uploaded audio file: {audio_file.filename}")
    try:
        # Prepare the file for forwarding
        files = {"audio_file": (audio_file.filename, audio_file.file, audio_file.content_type)}
        
        # Forward to Whisper service
        response = requests.post(
            f"{WHISPER_URL}/transcribe",
            files=files,
            timeout=120  # 2 minutes for transcription
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Transcription successful: '{result.get('text', '')[:100]}...'")
            return result
        else:
            logger.error(f"Whisper service error: {response.status_code}")
            return {"error": f"Whisper service error: {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return {"error": str(e)}

@app.api_route("/audio/{filename}", methods=["GET", "HEAD"])
async def proxy_audio(request: Request, filename: str, range: Optional[str] = Header(None)):
    """Proxy audio files from TTS services (Orpheus or Bark) for direct playback with range support"""
    logger.info(f"Proxying audio file: {filename}, Method: {request.method}, Range: {range}")

    try:
        # Determine which TTS service based on filename pattern
        # Bark files start with "bark_", Orpheus files don't
        tts_url = BARK_URL if filename.startswith("bark_") else ORPHEUS_URL
        tts_name = "Bark" if filename.startswith("bark_") else "Orpheus"
        logger.info(f"Routing {filename} to {tts_name} TTS service")
        
        # For HEAD requests, we need to make a GET request to get headers, then return only headers
        if request.method == "HEAD":
            # Make a GET request to get file info
            response = requests.get(
                f"{tts_url}/audio/{filename}",
                timeout=30
            )
            
            if response.status_code == 200:
                response_headers = {
                    "Content-Type": "audio/wav",
                    "Accept-Ranges": "bytes",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                    "Access-Control-Allow-Headers": "Range",
                    "Cache-Control": "public, max-age=3600",
                    "Content-Length": str(len(response.content))
                }
                
                logger.info(f"HEAD request for {filename}: size={len(response.content)}")
                return Response(status_code=200, headers=response_headers)
            else:
                return Response(status_code=404)
        
        # For GET requests
        headers = {}
        if range:
            headers["Range"] = range
            
        # Forward request to the appropriate TTS service
        response = requests.get(
            f"{tts_url}/audio/{filename}",
            headers=headers,
            timeout=30
        )
        
        if response.status_code in [200, 206]:  # 206 for partial content
            # Prepare response headers for HTML5 audio compatibility
            response_headers = {
                "Content-Type": response.headers.get("Content-Type", "audio/wav"),
                "Accept-Ranges": "bytes",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "Range, Content-Type",
                "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges",
                "Cache-Control": "public, max-age=3600"
            }
            
            # Copy relevant headers from upstream response
            if "content-length" in response.headers:
                response_headers["Content-Length"] = response.headers["content-length"]
            if "content-range" in response.headers:
                response_headers["Content-Range"] = response.headers["content-range"]
                
            status_code = response.status_code
            logger.info(f"Successfully proxied audio file: {filename} (status: {status_code}, size: {response_headers.get('Content-Length', 'unknown')} bytes)")
            
            return Response(
                content=response.content,
                status_code=status_code,
                headers=response_headers,
                media_type="audio/wav"
            )
        else:
            logger.error(f"Audio file not found: {filename} (status: {response.status_code})")
            return {"error": "Audio file not found"}
            
    except Exception as e:
        logger.error(f"Error proxying audio file {filename}: {e}")
        return {"error": str(e)}