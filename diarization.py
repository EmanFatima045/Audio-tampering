import os
import torch
import librosa
import torchaudio
import soundfile as sf
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from speechbrain.inference.speaker import SpeakerRecognition



# Filter out specific warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings("ignore", message=".*Requested Pretrainer collection using symlinks on Windows.*")

# Pretrained models
diarization_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"}  # Force CPU to avoid CUDA warnings if not needed
)


def convert_to_mono_16k(input_path: str, output_path: str = "converted_audio.wav") -> str:
    import subprocess

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}")

    return output_path


def run_diarization(file_path: str) -> dict:
    file_path = convert_to_mono_16k(file_path)
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

    speaker_segments = {}

    for label, (start, end) in zip(labels, timestamps):
        if label not in speaker_segments:
            speaker_segments[label] = []
        speaker_segments[label].append((round(start, 2), round(end, 2)))

    results = []
    for label, segments in speaker_segments.items():
        merged_segments = merge_adjacent_segments(segments)
        results.append({
            "speaker": f"Speaker {label}",
            "segments": merged_segments
        })

    return {
        "estimated_speakers": best_k,
        "grouped_diarization": results
    }


def merge_adjacent_segments(segments, gap_threshold=0.5):
    """Merge adjacent segments if they are close enough"""
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x[0])
    
    merged = [sorted_segments[0]]
    
    for current in sorted_segments[1:]:
        previous = merged[-1]
        
        # If current segment starts soon after previous ends, merge them
        if current[0] - previous[1] <= gap_threshold:
            merged[-1] = (previous[0], current[1])  # Extend previous segment
        else:
            merged.append(current)  # Add as new segment
            
    return merged


if __name__ == "__main__":
    file_path = r"C:\Users\Dr Bia\Desktop\sample Repository\audio_2.mp3" 
    result = run_diarization(file_path)
    print(result)


