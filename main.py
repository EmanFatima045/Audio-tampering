from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import base64
import tempfile
import os
from matplotlib import pyplot as plt
from io import BytesIO
from datetime import datetime
from pathlib import Path

from transcribe import transcribe_audio
from sentiment_analysis import get_sentiment
from temporal_inconsistency import detect_splices_by_background, detect_phase_mismatch, analyze_audio_splices, plot_combined_analysis_base64
from metadata import extract_audio_metadata
from diarization import run_diarization
from gender_detection import process_audio

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

@app.post("/gender/")
async def gender_endpoint(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        file_bytes = await file.read()
        gender = process_audio(file_bytes, file.filename)

        return {"gender": gender}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/temporal_inconsistency/")
async def temporal_inconsistency_endpoint(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        try:
            # Run the analysis
            bg_res, phase_res, high_confidence_splices = analyze_audio_splices(temp_path)

            # Convert background detections
            background_splices = [
                {"time": float(time), "confidence": float(conf)}
                for time, conf in zip(bg_res['times'], bg_res.get('confidence', []))
            ]

            # Convert phase detections
            phase_splices = [
                {"time": float(time), "confidence": float(conf)}
                for time, conf in zip(phase_res['times'], phase_res.get('confidence', []))
            ]

            # Combined detections
            combined_splices = [
                {"time": float(splice['time']), 
                 "confidence": float(splice['confidence']), 
                 "methods": splice['methods']}
                for splice in high_confidence_splices
            ]

            # Generate plot as base64
            graph_base64 = None
            try:
                graph_base64 = plot_combined_analysis_base64(bg_res, phase_res, high_confidence_splices, temp_path)
            except Exception as plot_error:
                print(f"Error generating plot: {plot_error}")
                graph_base64 = None

            # Build JSON response
            return {
                "file": file.filename,
                "background_splices": background_splices,
                "phase_splices": phase_splices,
                "combined_splices": combined_splices,
                "graph": graph_base64,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
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
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metadata/")
async def comprehensive_metadata_endpoint(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".aiff")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        try:
            # Run the comprehensive metadata analysis
            result = extract_audio_metadata(temp_path)
            
            if not result["success"]:
                raise HTTPException(status_code=400, detail=result["error"])
            
            # Prepare the response data
            metadata = result["metadata"]
            
            # Convert any non-serializable data to strings
            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {key: make_json_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                else:
                    return str(obj)
            
            serializable_metadata = make_json_serializable(metadata)
            
            return {
                "success": True,
                "filename": file.filename,
                "analysis_timestamp": datetime.now().isoformat(),
                "metadata": serializable_metadata
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Metadata analysis error: {str(e)}")
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))