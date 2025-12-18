"""
NISQA-TTS Service - FastAPI wrapper for NISQA-TTS MOS prediction

NISQA-TTS is optimized for predicting Mean Opinion Score (MOS) of synthetic speech.
It returns overall MOS (optional: plus 4 quality dimensions: noisiness, discontinuity, coloration, loudness)

Model: https://github.com/gabrielmittag/NISQA
Paper: https://arxiv.org/abs/2104.09494
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import logging
import os
import sys
import torch
import soundfile as sf

# Add NISQA to Python path
NISQA_PATH = os.path.join(os.path.dirname(__file__), 'NISQA')
if os.path.exists(NISQA_PATH):
    sys.path.insert(0, NISQA_PATH)
else:
    raise RuntimeError(f"NISQA repository not found at {NISQA_PATH}. Run: git clone https://github.com/gabrielmittag/NISQA.git")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MediTalk NISQA-TTS MOS Prediction", version="1.0.0")

# Global model variable
nisqa_model = None


class MOSPredictionRequest(BaseModel):
    """Request model for MOS prediction from file path."""
    audio_path: str


class MOSPredictionResponse(BaseModel):
    """Response model with MOS score."""
    mos: float  # Overall MOS (1-5 scale)


@app.on_event("startup")
async def startup_event():
    """Load NISQA-TTS model on startup."""
    global nisqa_model
    try:
        logger.info("Loading NISQA-TTS model...")
        
        # Import NISQA
        from nisqa.NISQA_model import nisqaModel
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Path to pretrained NISQA-TTS model
        pretrained_model = os.path.join(
            os.path.dirname(__file__),
            'NISQA', 'weights', 'nisqa_tts.tar'
        )
        
        if not os.path.exists(pretrained_model):
            raise FileNotFoundError(f"NISQA-TTS model not found at {pretrained_model}")
        
        # Create a dummy audio file path for initialization
        # We'll update it per-request during prediction
        dummy_path = os.path.join(os.path.dirname(__file__), 'NISQA', 'weights', 'nisqa_tts.tar')
        
        # Initialize NISQA model with required args
        # We use a dummy file path and will reload dataset per prediction
        args = {
            'mode': 'predict_file',
            'pretrained_model': pretrained_model,
            'deg': dummy_path,  # Dummy path to avoid None error
            'output_dir': '/tmp',  # Will be overridden per request
            'tr_bs_val': 1,
            'tr_num_workers': 0,
            'ms_channel': None,  # For stereo files, which channel to use
        }
        
        nisqa_model = nisqaModel(args)
        
        logger.info("NISQA-TTS model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load NISQA-TTS model: {e}")
        logger.error("Make sure NISQA is installed and weights are available")
        import traceback
        traceback.print_exc()
        nisqa_model = None


@app.get("/health")
def health_check():
    """Health check endpoint."""
    if nisqa_model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model": "nisqa-tts",
                "error": "Model not loaded"
            }
        )
    return {
        "status": "healthy",
        "model": "nisqa-tts",
        "description": "MOS prediction for synthetic speech",
        "output_range": "1-5 (1=bad, 5=excellent)"
    }


@app.post("/predict_from_path", response_model=MOSPredictionResponse)
async def predict_from_path(request: MOSPredictionRequest):
    """
    Predict MOS quality score from audio file path.
    
    Args:
        request: Contains audio_path (absolute path to audio file)
        
    Returns:
        MOS scores: overall + 4 dimensions (noisiness, discontinuity, coloration, loudness)
    """
    if nisqa_model is None:
        raise HTTPException(status_code=503, detail="NISQA model not loaded")
    
    audio_path = request.audio_path
    
    # Check if file exists
    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {audio_path}"
        )
    
    try:
        logger.info(f"Predicting MOS for: {audio_path}")
        
        # Create temporary output directory for NISQA results
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Update args for this prediction
            nisqa_model.args['deg'] = audio_path
            nisqa_model.args['output_dir'] = tmp_dir
            
            # Reload dataset with new file
            nisqa_model._loadDatasets()
            
            # Run prediction
            nisqa_model.predict()
            
            # Read results from CSV
            import pandas as pd
            results_csv = os.path.join(tmp_dir, 'NISQA_results.csv')
            
            if not os.path.exists(results_csv):
                raise Exception("NISQA did not generate results file")
            
            df = pd.read_csv(results_csv)
            
            # Extract MOS score from first row
            mos = float(df['mos_pred'].iloc[0])
        
        logger.info(f"Prediction complete - MOS: {mos:.2f}")
        
        return MOSPredictionResponse(mos=mos)
        
    except Exception as e:
        logger.error(f"MOS prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"MOS prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    logger.info("Starting NISQA-TTS FastAPI service...")
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv('NISQA_PORT', 8006))
    
    logger.info(f"Starting NISQA-TTS service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
