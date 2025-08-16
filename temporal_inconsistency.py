import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage

# ========= Helper =========
def frame_params(sr, frame_ms=25, hop_ms=10):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    return frame_len, hop_len, frame_len

# ========= Background continuity detection =========
def detect_splices_by_background(audio_path, frame_ms=25, hop_ms=10, bg_percentile=30,
                                 window_frames=151, smooth_sigma=2,
                                 adapt_window=201, adapt_z=3.0, low_hz=300):
    y, sr = librosa.load(audio_path, sr=None)
    frame_len, hop_len, n_fft = frame_params(sr, frame_ms=frame_ms, hop_ms=hop_ms)

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_len)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_band_idx = np.where(freqs <= low_hz)[0]

    rms_low = np.sqrt(np.mean(np.abs(S[low_band_idx, :])**2, axis=0))

    bg_level = np.percentile(rms_low, bg_percentile)
    bg_series = pd.Series(rms_low).rolling(window=window_frames, center=True, min_periods=1).mean()
    smoothed = scipy.ndimage.gaussian_filter1d(bg_series.values, sigma=smooth_sigma)

    roll_mean = pd.Series(smoothed).rolling(window=adapt_window, center=True, min_periods=1).mean()
    roll_std = pd.Series(smoothed).rolling(window=adapt_window, center=True, min_periods=1).std()
    threshold = roll_mean + adapt_z * roll_std

    peaks = np.where(smoothed > threshold)[0]
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_len)

    return {
        "sr": sr,
        "smoothed": smoothed,
        "threshold": threshold.values,
        "peaks": peaks,
        "times": times,
        "hop_len": hop_len
    }

# ========= Phase mismatch detection =========
def detect_phase_mismatch(audio_path, frame_ms=25, hop_ms=10, z_thresh=3.0):
    y, sr = librosa.load(audio_path, sr=None)
    frame_len, hop_len, n_fft = frame_params(sr, frame_ms=frame_ms, hop_ms=hop_ms)

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_len)
    phase = np.angle(S)
    phase_unwrapped = np.unwrap(phase, axis=1)
    phase_diff = np.diff(phase_unwrapped, axis=1)
    mean_abs_diff = np.mean(np.abs(phase_diff), axis=0)

    norm_diff = (mean_abs_diff - np.mean(mean_abs_diff)) / (np.std(mean_abs_diff) + 1e-12)
    peaks = np.where(norm_diff > z_thresh)[0]
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_len)

    return {
        "sr": sr,
        "norm_diff": norm_diff,
        "peaks": peaks,
        "times": times,
        "hop_len": hop_len
    }

# ========= Combined plot =========
def plot_combined(bg_res, phase_res, audio_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ===== Background continuity subplot =====
    times_bg = librosa.frames_to_time(np.arange(len(bg_res['smoothed'])), 
                                      sr=bg_res['sr'], hop_length=bg_res['hop_len'])
    ax1.plot(times_bg, bg_res['smoothed'], label='Background Continuity', color='blue')
    ax1.plot(times_bg, bg_res['threshold'], '--', color='orange', label='BG Threshold')
    ax1.scatter(librosa.frames_to_time(bg_res['peaks'], sr=bg_res['sr'], hop_length=bg_res['hop_len']),
                bg_res['smoothed'][bg_res['peaks']], color='red', label='BG Splice')
    ax1.set_ylabel("Score")
    ax1.set_title("Background Continuity Analysis")
    ax1.legend()

    # ===== Phase mismatch subplot =====
    times_phase = librosa.frames_to_time(np.arange(len(phase_res['norm_diff'])), 
                                         sr=phase_res['sr'], hop_length=phase_res['hop_len'])
    phase_scaled = (phase_res['norm_diff'] - np.min(phase_res['norm_diff'])) / \
                   (np.max(phase_res['norm_diff']) - np.min(phase_res['norm_diff']))
    ax2.plot(times_phase, phase_scaled, label='Phase Mismatch (scaled)', color='green')
    ax2.scatter(librosa.frames_to_time(phase_res['peaks'], sr=phase_res['sr'], hop_length=phase_res['hop_len']),
                phase_scaled[phase_res['peaks']], color='purple', label='Phase Splice')

    # Annotate timestamps neatly
    peak_times = librosa.frames_to_time(phase_res['peaks'], sr=phase_res['sr'], hop_length=phase_res['hop_len'])
    for i, t in enumerate(peak_times):
        if i % 3 == 0:  # label every 3rd peak to avoid clutter
            ax2.annotate(f"{t:.2f}", (t, phase_scaled[phase_res['peaks'][i]]),
                         textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Score")
    ax2.set_title("Phase Mismatch Analysis")
    ax2.legend()

    plt.suptitle(f"Audio Splice Detection: {audio_path}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



