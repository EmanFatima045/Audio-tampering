import os
import torch
import librosa
import torchaudio
import soundfile as sf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from speechbrain.inference.speaker import SpeakerRecognition
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Pretrained models
diarization_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

gender_model_name = "prithivMLmods/Common-Voice-Gender-Detection"
gender_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(gender_model_name)
gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(gender_model_name)
id2label = {"0": "Female", "1": "Male"}


def convert_to_mono_16k(input_path: str, output_path: str = "converted_audio.wav") -> str:
    import subprocess

    # Ensure input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    #  output to mono 16kHz WAV using ffmpeg
    command = [
        "ffmpeg",  # make sure ffmpeg is in PATH or give full path here
        "-y",  # overwrite output file if exists
        "-i", input_path,
        "-ac", "1",           # mono
        "-ar", "16000",       # 16kHz
        "-vn",                # no video
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}")

    return output_path




def detect_gender(audio_chunk: torch.Tensor) -> str:
    tmp_path = "tmp_chunk.wav"
    sf.write(tmp_path, audio_chunk.squeeze().numpy(), 16000)

    speech, sr = librosa.load(tmp_path, sr=16000)
    inputs = gender_feature_extractor(
        speech, sampling_rate=sr, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = gender_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
        predicted_label = torch.argmax(probs).item()
    os.remove(tmp_path)
    return id2label[str(predicted_label)]


def run_diarization_gender(file_path: str) -> dict:
    file_path = convert_to_mono_16k(file_path)  # Convert and resample
    signal, sr = torchaudio.load(file_path)

    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if sr != 16000:
        raise ValueError("Audio must be 16kHz.")

    chunk_len = int(sr * 1.0)
    chunks = [signal[:, i:i + chunk_len] for i in range(0, signal.shape[1], chunk_len) if (i + chunk_len) <= signal.shape[1]]

    embeddings = []
    timestamps = []
    chunk_duration = chunk_len / sr

    for idx, chunk in enumerate(chunks):
        emb = diarization_model.encode_batch(chunk).squeeze().detach().numpy()
        embeddings.append(emb)
        timestamps.append((idx * chunk_duration, (idx + 1) * chunk_duration))

    X = np.array(embeddings)
    best_k = 2
    best_score = -1

    for k in range(2, min(len(X), 8) + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k

    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(X)
    labels = kmeans.labels_

    results = []
    for label, chunk, (start, end) in zip(labels, chunks, timestamps):
        gender = detect_gender(chunk)
        results.append({
            "speaker": f"Speaker {label}",
            "gender": gender,
            "start": round(start, 2),
            "end": round(end, 2)
        })

    return {
        "estimated_speakers": best_k,
        "diarization_with_gender": results
    }
if __name__ == "__main__":
    file_path = r"C:\Users\Dr Bia\Desktop\sample Repository\audio_2.mp3" #file
    result = run_diarization_gender(file_path)
    print(result)


