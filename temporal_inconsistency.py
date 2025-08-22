import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage
from scipy.signal import find_peaks
from scipy import stats
import base64
from io import BytesIO

# ========= Helper Functions =========
def frame_params(sr, frame_ms=25, hop_ms=10):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    return frame_len, hop_len, frame_len

# Removed print_detection_summary function to eliminate console output

# ========= Enhanced Background Continuity Detection =========
def detect_splices_by_background(audio_path, frame_ms=25, hop_ms=10, bg_percentile=20,
                                 window_frames=101, smooth_sigma=1.5,
                                 adapt_window=151, adapt_z=2.5, low_hz=500,
                                 min_prominence=0.1, min_distance=20):
    """Enhanced background continuity detection with better peak finding"""
    
    y, sr = librosa.load(audio_path, sr=None)
    
    frame_len, hop_len, n_fft = frame_params(sr, frame_ms=frame_ms, hop_ms=hop_ms)

    # Compute STFT and extract low frequency energy
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_len)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_band_idx = np.where(freqs <= low_hz)[0]

    # Calculate RMS energy in low frequency band
    rms_low = np.sqrt(np.mean(np.abs(S[low_band_idx, :])**2, axis=0))
    
    # Enhanced background level estimation using multiple percentiles
    bg_level = np.percentile(rms_low, bg_percentile)
    
    # Apply rolling window smoothing
    bg_series = pd.Series(rms_low).rolling(window=window_frames, center=True, min_periods=1).mean()
    smoothed = scipy.ndimage.gaussian_filter1d(bg_series.values, sigma=smooth_sigma)

    # Adaptive threshold calculation with improved statistics
    roll_mean = pd.Series(smoothed).rolling(window=adapt_window, center=True, min_periods=1).mean()
    roll_std = pd.Series(smoothed).rolling(window=adapt_window, center=True, min_periods=1).std()
    roll_mad = pd.Series(smoothed).rolling(window=adapt_window, center=True, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x))))
    
    # Use MAD-based threshold for robustness
    threshold = roll_mean + adapt_z * (roll_mad * 1.4826)  # 1.4826 converts MAD to std equivalent
    
    # Find peaks with minimum prominence and distance
    peaks, properties = find_peaks(smoothed, 
                                   height=threshold.values,
                                   prominence=min_prominence,
                                   distance=min_distance)
    
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_len)
    
    # Calculate confidence scores
    confidence_scores = smoothed[peaks] - threshold.values[peaks]
    
    return {
        "sr": sr,
        "smoothed": smoothed,
        "threshold": threshold.values,
        "peaks": peaks,
        "times": times,
        "hop_len": hop_len,
        "confidence": confidence_scores,
        "bg_level": bg_level
    }

# ========= Enhanced Phase Mismatch Detection =========
def detect_phase_mismatch(audio_path, frame_ms=25, hop_ms=10, z_thresh=2.8,
                         freq_range=(200, 4000), min_prominence=0.3, min_distance=15):
    """Enhanced phase mismatch detection with frequency filtering"""
    
    y, sr = librosa.load(audio_path, sr=None)
    frame_len, hop_len, n_fft = frame_params(sr, frame_ms=frame_ms, hop_ms=hop_ms)

    # Compute STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_len)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Focus on specific frequency range for phase analysis
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    S_filtered = S[freq_mask, :]
    
    # Calculate phase differences
    phase = np.angle(S_filtered)
    phase_unwrapped = np.unwrap(phase, axis=1)
    
    # Multiple phase difference metrics
    phase_diff = np.diff(phase_unwrapped, axis=1)
    phase_diff_2nd = np.diff(phase_diff, axis=1)  # Second derivative for discontinuities
    
    # Combine metrics
    mean_abs_diff = np.mean(np.abs(phase_diff), axis=0)
    mean_abs_diff_2nd = np.mean(np.abs(phase_diff_2nd), axis=0)
    
    # Weighted combination
    combined_metric = 0.7 * mean_abs_diff[1:] + 0.3 * mean_abs_diff_2nd
    
    # Robust normalization using median and MAD
    median_val = np.median(combined_metric)
    mad_val = np.median(np.abs(combined_metric - median_val))
    norm_diff = (combined_metric - median_val) / (mad_val * 1.4826 + 1e-12)
    
    # Find peaks with improved parameters
    peaks, properties = find_peaks(norm_diff, 
                                   height=z_thresh,
                                   prominence=min_prominence,
                                   distance=min_distance)
    
    # Adjust peak indices for frame alignment
    peaks = peaks + 1  # Compensate for diff operation
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_len)
    
    # Calculate confidence scores
    confidence_scores = norm_diff[peaks - 1] if len(peaks) > 0 else []
    
    return {
        "sr": sr,
        "norm_diff": norm_diff,
        "peaks": peaks,
        "times": times,
        "hop_len": hop_len,
        "confidence": confidence_scores,
        "combined_metric": combined_metric
    }

# ========= Combined Analysis (Background + Phase Only) =========
def analyze_audio_splices(audio_path, confidence_threshold=0.6):
    """Comprehensive splice detection analysis using background and phase methods"""
    
    # Run detection methods (removing spectral analysis)
    bg_res = detect_splices_by_background(audio_path)
    phase_res = detect_phase_mismatch(audio_path)
    
    # Combine results with confidence weighting
    all_detections = []
    
    # Add background detections
    for i, (time, conf) in enumerate(zip(bg_res['times'], bg_res.get('confidence', []))):
        all_detections.append({
            'time': time,
            'method': 'Background',
            'confidence': conf,
            'frame': bg_res['peaks'][i]
        })
    
    # Add phase detections
    for i, (time, conf) in enumerate(zip(phase_res['times'], phase_res.get('confidence', []))):
        all_detections.append({
            'time': time,
            'method': 'Phase',
            'confidence': conf,
            'frame': phase_res['peaks'][i]
        })
    
    # Sort by time and group nearby detections
    all_detections.sort(key=lambda x: x['time'])
    
    # Group detections within 0.5 seconds
    grouped_detections = []
    if all_detections:
        current_group = [all_detections[0]]
        
        for detection in all_detections[1:]:
            if detection['time'] - current_group[-1]['time'] <= 0.5:
                current_group.append(detection)
            else:
                grouped_detections.append(current_group)
                current_group = [detection]
        grouped_detections.append(current_group)
    
    high_confidence_splices = []
    
    for group in grouped_detections:
        avg_time = np.mean([d['time'] for d in group])
        methods = [d['method'] for d in group]
        confidences = [d['confidence'] for d in group]
        
        # Calculate group confidence
        method_score = len(set(methods)) / 2.0  # Bonus for both methods agreeing
        avg_confidence = np.mean(confidences)
        combined_confidence = method_score * 0.4 + (avg_confidence / max(confidences)) * 0.6
        
        if combined_confidence >= confidence_threshold:
            high_confidence_splices.append({
                'time': avg_time,
                'confidence': combined_confidence,
                'methods': list(set(methods))
            })
    
    return bg_res, phase_res, high_confidence_splices

def plot_combined_analysis_base64(bg_res, phase_res, high_confidence_splices, audio_path):
    """Enhanced plotting with background and phase detection methods - returns base64"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    # ===== Background continuity subplot =====
    times_bg = librosa.frames_to_time(np.arange(len(bg_res['smoothed'])), 
                                      sr=bg_res['sr'], hop_length=bg_res['hop_len'])
    ax1.plot(times_bg, bg_res['smoothed'], label='Background Energy', color='blue', linewidth=1.5)
    ax1.plot(times_bg, bg_res['threshold'], '--', color='orange', label='Threshold', linewidth=1)
    
    if len(bg_res['peaks']) > 0:
        ax1.scatter(bg_res['times'], bg_res['smoothed'][bg_res['peaks']], 
                   color='red', s=50, label='BG Detections', zorder=5)
    
    ax1.axhline(y=bg_res['bg_level'], color='gray', linestyle=':', alpha=0.7, label='Base Level')
    ax1.set_ylabel("Energy")
    ax1.set_title("Background Continuity Analysis")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== Phase mismatch subplot =====
    times_phase = librosa.frames_to_time(np.arange(len(phase_res['norm_diff'])), 
                                         sr=phase_res['sr'], hop_length=phase_res['hop_len'])
    ax2.plot(times_phase, phase_res['norm_diff'], label='Phase Discontinuity', color='green', linewidth=1.5)
    ax2.axhline(y=2.8, color='orange', linestyle='--', label='Threshold', linewidth=1)
    
    if len(phase_res['peaks']) > 0:
        phase_peak_times = librosa.frames_to_time(phase_res['peaks'], 
                                                sr=phase_res['sr'], hop_length=phase_res['hop_len'])
        ax2.scatter(phase_peak_times, phase_res['norm_diff'][phase_res['peaks'] - 1], 
                   color='purple', s=50, label='Phase Detections', zorder=5)
    
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Z-Score")
    ax2.set_title("Phase Mismatch Analysis")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Highlight high confidence splices across both subplots
    for splice in high_confidence_splices:
        for ax in [ax1, ax2]:
            ax.axvline(x=splice['time'], color='red', linestyle='-', alpha=0.8, linewidth=2)
            ax.text(splice['time'], ax.get_ylim()[1] * 0.9, 
                   f"SPLICE {splice['confidence']:.2f}", 
                   rotation=90, ha='right', va='top', fontweight='bold', color='red')
    
    plt.suptitle(f"Audio Splice Detection Analysis: {audio_path}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Convert to base64 instead of showing
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()  # Important: close the figure to free memory
    buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode("utf-8")
    
    return graph_base64