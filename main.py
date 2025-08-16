from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import io
import base64
import tempfile
import os
from matplotlib import pyplot as plt

from transcribe import transcribe_audio
from sentiment_analysis import get_sentiment
from temporal_inconsistency import detect_splices_by_background, detect_phase_mismatch, plot_combined

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe/")
async def transcribe_endpoint(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        file_bytes = await file.read()
        transcription = transcribe_audio(file_bytes, file.filename)

        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment/")
async def sentiment_endpoint(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        file_bytes = await file.read()
        transcription = transcribe_audio(file_bytes, file.filename)
        sentiment = get_sentiment(transcription)

        return {
            "transcription": transcription,
            "sentiment": sentiment
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-temporal-inconsistency/")
async def detect_temporal_inconsistency_endpoint(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Read the uploaded file
        file_bytes = await file.read()
        
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Perform background continuity detection
            bg_results = detect_splices_by_background(temp_file_path)
            
            # Perform phase mismatch detection
            phase_results = detect_phase_mismatch(temp_file_path)
            
            # Generate the combined plot
            plt.ioff()  # Turn off interactive mode
            plot_combined(bg_results, phase_results, file.filename)
            
            # Save plot to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Convert to base64 for JSON response
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            plt.close()  # Close the plot to free memory
            
            # Prepare console-like output
            console_output = f"Background splice times: {bg_results['times']}\n"
            console_output += f"Phase mismatch times: {phase_results['times']}"
            
            return JSONResponse({
                "text_output": console_output,
                "graph_base64": img_base64
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        # Clean up temporary file in case of error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# Diarization Endpoint
@app.post("/diarization/")
async def diarization_endpoint(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await file.read())
            temp_path = temp_audio.name

        
        # Process diarization
        results = run_diarization(temp_path)

