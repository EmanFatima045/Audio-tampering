from fastapi import FastAPI, UploadFile, File
import soundfile as sf
import noisereduce as nr
import whisper
import os
import uuid
import psycopg2
import librosa
from datetime import datetime
import uvicorn

# FastAPI app
app = FastAPI()

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

# Run the server (for manual execution)
if __name__ == "__main__":
    uvicorn.run("RealDatabase:app", host="127.0.0.1", port=8000, reload=True)


