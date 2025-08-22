from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Form

import base64
import tempfile
import os
from matplotlib import pyplot as plt
from io import BytesIO
from datetime import datetime
from pathlib import Path
import uuid

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
async def transcribe(file: UploadFile = File(...)):
    try:
        # Check file type (basic)
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an audio file.")

        file_bytes = await file.read()
        transcription = transcribe_audio(file_bytes, file.filename)
        return {"transcription": transcription}

    except HTTPException as e:
        raise e  # re-raise FastAPI handled error

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/sentiment/")
async def sentiment_endpoint(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Read file once
        file_bytes = await file.read()

        # Optional: save to temp if you need a physical file
        with tempfile.NamedTemporaryFile(delete=True, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        try:
            # Run transcription on bytes
            transcription = transcribe_audio(file_bytes, file.filename)

            # Sentiment analysis on transcription
            sentiment = get_sentiment(transcription)

            return {
                "sentiment": sentiment
            }

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


@app.post("/gender/")
async def detect_gender(file: UploadFile = File(...)):
    try:
        # Check file type
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an audio file.")

        # Save temp file
        temp_id = str(uuid.uuid4())
        temp_path = f"{temp_id}.wav"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Process audio
        result = process_audio(temp_path)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/temporal_inconsistency/")
async def temporal_inconsistency_endpoint(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
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

        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio:
            temp_audio.write(await file.read())
            temp_path = temp_audio.name

        
        # Process diarization
        results = run_diarization(temp_path)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metadata/")
async def comprehensive_metadata_endpoint(
    file: UploadFile = File(...),
    original_modified: str = Form(None),
    original_created: str = Form(None)
):
    try:
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".aiff")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete= True, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        try:
            # Parse timestamps
            original_timestamps = {}
            if original_modified:
                try:
                    if original_modified.isdigit():
                        original_timestamps['modified'] = int(original_modified) / 1000
                    else:
                        dt = datetime.fromisoformat(original_modified.replace('Z', '+00:00'))
                        original_timestamps['modified'] = dt.timestamp()
                except:
                    original_timestamps['modified'] = None
            if original_created:
                try:
                    if original_created.isdigit():
                        original_timestamps['created'] = int(original_created) / 1000
                    else:
                        dt = datetime.fromisoformat(original_created.replace('Z', '+00:00'))
                        original_timestamps['created'] = dt.timestamp()
                except:
                    original_timestamps['created'] = None

            # Run metadata extractor
            result = extract_audio_metadata(
                filepath=temp_path,
                original_filename=file.filename,
                original_timestamps=original_timestamps if original_timestamps else None
            )

            if not result["success"]:
                raise HTTPException(status_code=400, detail=result["error"])

            return {
                "success": True,
                "filename": file.filename,
                "analysis_timestamp": datetime.now().isoformat(),
                "original_timestamps_received": {
                    "modified": original_modified,
                    "created": original_created
                } if (original_modified or original_created) else None,
                "metadata": result["metadata"]
            }

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata analysis error: {str(e)}")
