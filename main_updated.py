from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
from transcribe import transcribe_audio
from sentiment_analysis import get_sentiment
from diarization import run_diarization
from gender_detection import process_audio
from temporal_inconsistency import analyze_audio_splices, plot_combined_analysis_base64
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

import os
from pathlib import Path
from metadata import extract_audio_metadata
import io
import base64
import matplotlib.pyplot as plt
from datetime import datetime

# Database imports
from connection import get_db, create_tables
from services import AudioForensicsService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_tables()
    yield
    # Shutdown (if needed)

app = FastAPI(lifespan=lifespan)

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for diarization segments
os.makedirs("static/segments", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Case Management Endpoints
@app.post("/cases/")
async def create_case(
    file: UploadFile = File(...),
    name: str = Form(...),
    notes: str = Form(None),
    db: Session = Depends(get_db)
):
    """Create a new case and perform all analyses"""
    try:
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Read file content
        file_content = await file.read()
        
        # Create case in database
        service = AudioForensicsService(db)
        case = service.create_case(name, file.filename, file_content, notes)
        
        # Save file temporarily for analysis
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(file_content)
            temp_path = tmp.name

        try:
            # 1. Transcription
            transcription_text = transcribe_audio(file_content, file.filename)
            service.update_transcription(case.id, transcription_text)
            
            # 2. Sentiment Analysis
            sentiment_result = get_sentiment(transcription_text)
            service.update_sentiment(case.id, sentiment_result)
            
            # 3. Gender Detection
            gender_result = process_audio(temp_path)
            if isinstance(gender_result, dict):
                gender = gender_result.get("gender", str(gender_result))
            else:
                gender = str(gender_result)
            service.update_gender_detection(case.id, gender)
            
            # 4. Metadata Analysis
            metadata_result = extract_audio_metadata(
                filepath=temp_path,
                original_filename=file.filename
            )
            if metadata_result["success"]:
                service.update_metadata(case.id, metadata_result["metadata"])
            
            # 5. Temporal Inconsistency Analysis (no graph)
            bg_res, phase_res, high_confidence_splices = analyze_audio_splices(temp_path)
            
            background_splices = [
                {"time": float(time), "confidence": float(conf)}
                for time, conf in zip(bg_res['times'], bg_res.get('confidence', []))
            ]
            
            phase_splices = [
                {"time": float(time), "confidence": float(conf)}
                for time, conf in zip(phase_res['times'], phase_res.get('confidence', []))
            ]
            
            combined_splices = [
                {"time": float(splice['time']), 
                 "confidence": float(splice['confidence']), 
                 "methods": splice['methods']}
                for splice in high_confidence_splices
            ]
            
            service.update_temporal_analysis(case.id, background_splices, phase_splices, combined_splices)
            
            # 6. Diarization
            diarization_results = run_diarization(
                temp_path,
                segments_dir="static/segments",
                public_base="/static/segments"
            )
            
            segments_data = []
            for segment in diarization_results.get('segments', []):
                segments_data.append({
                    'speaker': segment['speaker'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'file_url': segment['file_url'],
                    'transcription': None,  # Will be filled by segment analysis
                    'sentiment': None,      # Will be filled by segment analysis
                    'gender': None          # Will be filled by segment analysis
                })
            
            service.update_diarization(
                case.id, 
                diarization_results.get('estimated_speakers', 0), 
                segments_data
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return {
            "id": case.id,
            "name": case.case_name,
            "created_at": case.created_at.isoformat(),
            "message": "Case created successfully with all analyses completed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cases/")
async def get_cases(db: Session = Depends(get_db)):
    """Get all cases"""
    service = AudioForensicsService(db)
    cases = service.get_all_cases()
    
    return [
        {
            "id": case.id,
            "name": case.case_name,
            "original_filename": case.original_filename,
            "created_at": case.created_at.isoformat(),
            "updated_at": case.updated_at.isoformat(),
            "notes": case.notes
        }
        for case in cases
    ]

@app.get("/cases/{case_id}")
async def get_case(case_id: str, db: Session = Depends(get_db)):
    """Get case with all analysis results"""
    service = AudioForensicsService(db)
    case = service.get_case(case_id)
    
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    result = {
        "id": case.id,
        "name": case.case_name,
        "original_filename": case.original_filename,
        "created_at": case.created_at.isoformat(),
        "updated_at": case.updated_at.isoformat(),
        "notes": case.notes,
        "analyses": {}
    }
    
    # Add transcription
    if case.transcription_text:
        result["analyses"]["transcription"] = {
            "text": case.transcription_text,
            "confidence": case.transcription_confidence,
            "language": case.transcription_language
        }
    
    # Add sentiment
    if case.sentiment_result:
        result["analyses"]["sentiment"] = {
            "sentiment": case.sentiment_result,
            "confidence": case.sentiment_confidence
        }
    
    # Add gender detection
    if case.gender_result:
        result["analyses"]["gender"] = {
            "gender": case.gender_result,
            "confidence": case.gender_confidence
        }
    
    # Add metadata
    if case.metadata_json:
        result["analyses"]["metadata"] = {
            "metadata": case.metadata_json,
            "original_timestamps": case.original_timestamps
        }
    
    # Add temporal analysis
    if case.combined_splices:
        result["analyses"]["temporal"] = {
            "background_splices": case.background_splices,
            "phase_splices": case.phase_splices,
            "combined_splices": case.combined_splices
        }
    
    # Add diarization
    if case.diarization_segments:
        result["analyses"]["diarization"] = {
            "estimated_speakers": case.estimated_speakers,
            "segments": case.diarization_segments
        }
    
    return result

@app.delete("/cases/{case_id}")
async def delete_case(case_id: str, db: Session = Depends(get_db)):
    """Delete a case and all associated data"""
    service = AudioForensicsService(db)
    success = service.delete_case(case_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Case not found")
    
    return {"message": "Case deleted successfully"}

# Individual Analysis Endpoints (for segment analysis)
@app.post("/cases/{case_id}/segments/{segment_index}/transcribe")
async def transcribe_segment(
    case_id: str, 
    segment_index: int, 
    db: Session = Depends(get_db)
):
    """Transcribe a specific diarization segment"""
    service = AudioForensicsService(db)
    case = service.get_case(case_id)
    
    if not case or not case.diarization_segments:
        raise HTTPException(status_code=404, detail="Case or segments not found")
    
    if segment_index >= len(case.diarization_segments):
        raise HTTPException(status_code=404, detail="Segment not found")
    
    segment = case.diarization_segments[segment_index]
    
    try:
        # Download segment file
        url = segment['file_url'] if segment['file_url'].startswith('http') else f"http://127.0.0.1:8000{segment['file_url']}"
        import requests
        response = requests.get(url)
        response.raise_for_status()
        
        # Transcribe
        transcription_text = transcribe_audio(response.content, f"segment_{segment_index}.wav")
        
        # Update segment
        service.update_segment_analysis(case_id, segment_index, transcription=transcription_text)
        
        return {"transcription": transcription_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cases/{case_id}/segments/{segment_index}/sentiment")
async def analyze_segment_sentiment(
    case_id: str, 
    segment_index: int, 
    db: Session = Depends(get_db)
):
    """Analyze sentiment for a specific diarization segment"""
    service = AudioForensicsService(db)
    case = service.get_case(case_id)
    
    if not case or not case.diarization_segments:
        raise HTTPException(status_code=404, detail="Case or segments not found")
    
    if segment_index >= len(case.diarization_segments):
        raise HTTPException(status_code=404, detail="Segment not found")
    
    segment = case.diarization_segments[segment_index]
    
    if not segment.get('transcription'):
        raise HTTPException(status_code=400, detail="Segment must be transcribed first")
    
    try:
        sentiment_result = get_sentiment(segment['transcription'])
        
        # Update segment
        service.update_segment_analysis(case_id, segment_index, sentiment=sentiment_result)
        
        return {"sentiment": sentiment_result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cases/{case_id}/segments/{segment_index}/gender")
async def detect_segment_gender(
    case_id: str, 
    segment_index: int, 
    db: Session = Depends(get_db)
):
    """Detect gender for a specific diarization segment"""
    service = AudioForensicsService(db)
    case = service.get_case(case_id)
    
    if not case or not case.diarization_segments:
        raise HTTPException(status_code=404, detail="Case or segments not found")
    
    if segment_index >= len(case.diarization_segments):
        raise HTTPException(status_code=404, detail="Segment not found")
    
    segment = case.diarization_segments[segment_index]
    
    try:
        # Download segment file
        url = segment['file_url'] if segment['file_url'].startswith('http') else f"http://127.0.0.1:8000{segment['file_url']}"
        import requests
        response = requests.get(url)
        response.raise_for_status()
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(response.content)
            temp_path = tmp.name
        
        try:
            gender_result = process_audio(temp_path)
            if isinstance(gender_result, dict):
                gender = gender_result.get("gender", str(gender_result))
            else:
                gender = str(gender_result)
            
            # Update segment
            service.update_segment_analysis(case_id, segment_index, gender=gender)
            
            return {"gender": gender}
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints (for backward compatibility with ExploreFunctionalities)
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
async def detect_gender(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await file.read())
            temp_path = temp_audio.name

        gender = process_audio(temp_path)
        if isinstance(gender, dict):
            gender = gender.get("gender", str(gender))
        else:
            gender = str(gender)

        return {"gender": gender}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diarization/")
async def diarization_endpoint(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(await file.read())
            temp_path = temp_audio.name

        try:
            results = run_diarization(
                temp_path,
                segments_dir="static/segments",
                public_base="/static/segments"
            )
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return JSONResponse(content=results)
    except HTTPException:
         raise
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

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        try:
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

@app.post("/temporal_inconsistency/")
async def temporal_inconsistency_endpoint(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        try:
            bg_res, phase_res, high_confidence_splices = analyze_audio_splices(temp_path)

            background_splices = [
                {"time": float(time), "confidence": float(conf)}
                for time, conf in zip(bg_res['times'], bg_res.get('confidence', []))
            ]

            phase_splices = [
                {"time": float(time), "confidence": float(conf)}
                for time, conf in zip(phase_res['times'], phase_res.get('confidence', []))
            ]

            combined_splices = [
                {"time": float(splice['time']), 
                 "confidence": float(splice['confidence']), 
                 "methods": splice['methods']}
                for splice in high_confidence_splices
            ]

            graph_base64 = None
            try:
                graph_base64 = plot_combined_analysis_base64(bg_res, phase_res, high_confidence_splices, temp_path)
            except Exception as plot_error:
                print(f"Error generating plot: {plot_error}")
                graph_base64 = None

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
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
