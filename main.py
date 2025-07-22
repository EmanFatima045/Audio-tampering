from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from transcribe import transcribe_audio
from sentiment_analysis import get_sentiment

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
