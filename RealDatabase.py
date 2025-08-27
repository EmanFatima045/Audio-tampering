from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import soundfile as sf
import noisereduce as nr
import whisper
import os
import uuid
import psycopg2
import librosa
from datetime import datetime
import uvicorn
import json

# Reuse existing analysis modules
from transcribe import transcribe_audio
from diarization_gender import run_diarization
from metadata import extract_audio_metadata
from temporal_inconsistency import analyze_audio_splices
from sentiment_analysis import get_sentiment
from gender_detection import process_audio

# FastAPI app
app = FastAPI()

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static for serving diarization segments
os.makedirs("static/segments", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Whisper model once
model = whisper.load_model("base")

# PostgreSQL DB connection
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="audio_forensics_db",
        user="postgres",
        password="eman123",
        port=5432
    )

def init_db():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                id UUID PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata JSONB,
                temporal_splices JSONB,
                original_filename TEXT,
                notes TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS case_speakers (
                id UUID PRIMARY KEY,
                case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
                speaker_label TEXT NOT NULL,
                segments JSONB NOT NULL,
                transcript TEXT,
                sentiment TEXT,
                gender TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

init_db()

# Transcription logic
def transcribe_audio(file_bytes, filename):
    try:
        print("🔍 Transcription started...")

        temp_id = str(uuid.uuid4())
        original_path = f"{temp_id}_orig.wav"
        cleaned_path = f"{temp_id}_clean.wav"

        # Save original audio
        with open(original_path, "wb") as f:
            f.write(file_bytes)
        print(" Audio saved:", original_path)

        # Preprocessing
        audio, sr = librosa.load(original_path, sr=None, mono=True)
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        trimmed_audio, _ = librosa.effects.trim(audio_resampled, top_db=20)
        reduced_noise = nr.reduce_noise(y=trimmed_audio, sr=16000)
        sf.write(cleaned_path, reduced_noise, 16000)

        # Whisper transcription
        result = model.transcribe(cleaned_path)
        transcript_text = result["text"]
        print("📝 Transcription:", transcript_text)

        # Store into PostgreSQL
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO transcription (audio_filename, transcription_text, created_at)
            VALUES (%s, %s, %s)
        """, (filename, transcript_text, datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
        print("Data stored in PostgreSQL.")

        # Clean temp files
        os.remove(original_path)
        os.remove(cleaned_path)

        return transcript_text

    except Exception as e:
        print(" Error:", str(e))
        raise e

# FastAPI route
@app.post("/transcribe/")
async def upload_audio(file: UploadFile = File(...)):
    file_bytes = await file.read()
    transcription = transcribe_audio(file_bytes, file.filename)
    return {
        "filename": file.filename,
        "transcription": transcription
    }

def _insert_case(conn, case_id, name, metadata_obj, temporal_obj, original_filename, notes):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO cases (id, name, metadata, temporal_splices, original_filename, notes)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (case_id, name, json.dumps(metadata_obj) if metadata_obj is not None else None,
         json.dumps(temporal_obj) if temporal_obj is not None else None,
         original_filename, notes)
    )
    cur.close()

def _insert_case_speaker(conn, row_id, case_id, label, segments_obj, transcript, sentiment, gender):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO case_speakers (id, case_id, speaker_label, segments, transcript, sentiment, gender)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (row_id, case_id, label, json.dumps(segments_obj), transcript, sentiment, gender)
    )
    cur.close()

@app.post("/cases")
async def create_case(
    file: UploadFile = File(...),
    name: str = Form(...),
    notes: str = Form(None)
):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Save upload to a temp file (preserve extension for ffmpeg)
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    temp_path = f"{uuid.uuid4().hex}{suffix}"
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Metadata
        metadata_result = extract_audio_metadata(
            filepath=temp_path,
            original_filename=file.filename,
            original_timestamps=None
        )
        metadata_obj = metadata_result.get("metadata") if isinstance(metadata_result, dict) else None

        # Temporal analysis (use combined splices only)
        bg_res, phase_res, combined_splices = analyze_audio_splices(temp_path)
        temporal_obj = [
            {"time": float(x.get("time")), "confidence": float(x.get("confidence")), "methods": x.get("methods", [])}
            for x in combined_splices
        ]

        # Diarization
        diar = run_diarization(temp_path, segments_dir="static/segments", public_base="/static/segments")
        estimated = int(diar.get("estimated_speakers", 0))
        segments_list = diar.get("segments", [])

        # Prepare DB
        case_id = uuid.uuid4()
        conn = get_db_connection()
        try:
            _insert_case(conn, case_id, name, metadata_obj, temporal_obj, file.filename, notes)

            if estimated <= 1:
                # Whole-audio transcription/sentiment/gender
                transcript_text = transcribe_audio(content, file.filename)
                sentiment = get_sentiment(transcript_text)
                gender = process_audio(temp_path).get("gender")

                # Segments: use diarization segments if present else single full
                if segments_list:
                    merged_segments = segments_list
                else:
                    merged_segments = [{"start": 0.0, "end": 0.0, "file_url": ""}]

                speaker_row_id = uuid.uuid4()
                _insert_case_speaker(
                    conn,
                    speaker_row_id,
                    case_id,
                    "SingleSpeaker",
                    merged_segments,
                    transcript_text,
                    sentiment,
                    gender
                )
            else:
                # Group by speaker label
                speakers = {}
                for seg in segments_list:
                    label = seg.get("speaker", "Unknown")
                    speakers.setdefault(label, {"segments": [], "transcripts": [], "sentiments": [], "genders": []})
                    speakers[label]["segments"].append({
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "file_url": seg.get("file_url")
                    })

                # For each speaker, transcribe each segment and derive sentiment/gender
                for label, data in speakers.items():
                    for seg in data["segments"]:
                        file_url = seg.get("file_url") or ""
                        # Convert public URL to local path
                        local_path = None
                        if file_url.startswith("/static/segments/"):
                            local_path = file_url.lstrip("/")
                        if local_path and os.path.exists(local_path):
                            with open(local_path, "rb") as f:
                                seg_bytes = f.read()
                            seg_transcript = transcribe_audio(seg_bytes, os.path.basename(local_path))
                            data["transcripts"].append(seg_transcript)
                            data["sentiments"].append(get_sentiment(seg_transcript))
                            # Optional: gender per segment; choose first non-null later
                            try:
                                g = process_audio(local_path).get("gender")
                                if g:
                                    data["genders"].append(g)
                            except Exception:
                                pass

                    # Aggregate per speaker
                    transcript_joined = "\n".join([t for t in data["transcripts"] if t]) if data["transcripts"] else None
                    # If multiple sentiments, pick the most frequent
                    sentiment_final = None
                    if data["sentiments"]:
                        from collections import Counter
                        sentiment_final = Counter(data["sentiments"]).most_common(1)[0][0]
                    gender_final = data["genders"][0] if data["genders"] else None

                    _insert_case_speaker(
                        conn,
                        uuid.uuid4(),
                        case_id,
                        label,
                        data["segments"],
                        transcript_joined,
                        sentiment_final,
                        gender_final
                    )

            conn.commit()
        finally:
            conn.close()

        return {"id": str(case_id), "name": name, "created_at": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

# Run the server (for manual execution)
if __name__ == "__main__":
    uvicorn.run("RealDatabase:app", host="127.0.0.1", port=8000, reload=True)


