import librosa
import soundfile as sf
import noisereduce as nr
import whisper
import os
import uuid

# Load Whisper model once
model = whisper.load_model("base")

def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    # Create temporary filenames
    temp_id = str(uuid.uuid4())
    original_path = f"{temp_id}_orig.wav"
    cleaned_path = f"{temp_id}_clean.wav"

    try:
        # Save original uploaded file
        with open(original_path, "wb") as f:
            f.write(file_bytes)

        # Load and process audio
        audio, sr = librosa.load(original_path, sr=None, mono=True)
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Trim silence and reduce noise
        trimmed_audio, _ = librosa.effects.trim(audio_resampled, top_db=20)
        reduced_noise = nr.reduce_noise(y=trimmed_audio, sr=16000)

        # Save cleaned audio
        sf.write(cleaned_path, reduced_noise, 16000)

        # Transcribe
        result = model.transcribe(cleaned_path)
        transcription = result["text"]

    finally:
        # Clean up temp files
        if os.path.exists(original_path):
            os.remove(original_path)
        if os.path.exists(cleaned_path):
            os.remove(cleaned_path)

    return transcription
