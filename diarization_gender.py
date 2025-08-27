import os
import uuid
import subprocess
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torchaudio
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from speechbrain.inference.speaker import SpeakerRecognition

# Speaker embedding model
diarization_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

def convert_to_mono_16k(input_path: str, output_path: str) -> str:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}")
    return output_path

def _chunk_signal(signal: torch.Tensor, sr: int, chunk_seconds: float = 1.0) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], float]:
    chunk_len = int(sr * chunk_seconds)
    total_len = signal.shape[1]
    chunks: List[torch.Tensor] = []
    frame_spans: List[Tuple[int, int]] = []
    for i in range(0, total_len, chunk_len):
        end = i + chunk_len
        if end > total_len:
            break
        chunks.append(signal[:, i:end])
        frame_spans.append((i, end))
    return chunks, frame_spans, chunk_seconds

def _merge_consecutive(labels: np.ndarray, frame_spans: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    merged: List[Tuple[int, int, int]] = []
    if len(labels) == 0:
        return merged
    cur_label = int(labels[0])
    cur_start = frame_spans[0][0]
    cur_end = frame_spans[0][1]
    for idx in range(1, len(labels)):
        if int(labels[idx]) == cur_label and frame_spans[idx][0] == cur_end:
            cur_end = frame_spans[idx][1]
        else:
            merged.append((cur_label, cur_start, cur_end))
            cur_label = int(labels[idx])
            cur_start = frame_spans[idx][0]
            cur_end = frame_spans[idx][1]
    merged.append((cur_label, cur_start, cur_end))
    return merged

def run_diarization(file_path: str, segments_dir: str = "static/segments", public_base: str = "/static/segments") -> dict:
    os.makedirs(segments_dir, exist_ok=True)

    converted_tmp = os.path.join(segments_dir, f"tmp_{uuid.uuid4().hex}.wav")
    converted = convert_to_mono_16k(file_path, output_path=converted_tmp)
    signal, sr = torchaudio.load(converted)

    try:
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        if sr != 16000:
            raise ValueError("Audio must be 16kHz mono after conversion")

        chunks, frame_spans, _ = _chunk_signal(signal, sr, chunk_seconds=1.0)
        if len(chunks) == 0:
            return {"estimated_speakers": 0, "segments": []}

        # Embeddings
        X_list: List[np.ndarray] = []
        for chunk in chunks:
            with torch.no_grad():
                emb = diarization_model.encode_batch(chunk).squeeze().detach().cpu().numpy()
            X_list.append(emb)
        X = np.array(X_list)

        # Choose K
        if len(X) == 1:
            best_k = 1
        else:
            best_score = -1.0
            best_k = 2
            for k in range(2, min(len(X), 8) + 1):
                kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(X)
                try:
                    score = silhouette_score(X, kmeans.labels_)
                except Exception:
                    score = -1.0
                if score > best_score:
                    best_score = score
                    best_k = k

        kmeans = KMeans(n_clusters=best_k, random_state=0, n_init='auto').fit(X) if best_k > 1 else None
        labels = kmeans.labels_ if kmeans is not None else np.zeros(len(X), dtype=int)

        # Merge adjacent same-speaker chunks
        merged = _merge_consecutive(labels, frame_spans)

        segments: List[Dict[str, Any]] = []
        for (speaker_label, start_frame, end_frame) in merged:
            start_sec = round(start_frame / sr, 2)
            end_sec = round(end_frame / sr, 2)

            segment_tensor = signal[:, start_frame:end_frame]
            seg_name = f"{uuid.uuid4().hex}.wav"
            seg_path = os.path.join(segments_dir, seg_name)

            sf.write(seg_path, segment_tensor.squeeze().cpu().numpy(), 16000)

            segments.append({
                "speaker": f"Speaker {speaker_label}",
                "start": start_sec,
                "end": end_sec,
                "file_url": f"{public_base}/{seg_name}"
            })

        return {"estimated_speakers": int(best_k), "segments": segments}
    finally:
        try:
            if os.path.exists(converted):
                os.remove(converted)
        except Exception:
            pass