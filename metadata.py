import os
from pathlib import Path
from datetime import datetime
import hashlib
import struct

try:
    from mutagen import File
    import librosa
    import numpy as np
    from scipy import stats
except ImportError:
    raise ImportError("Required libraries: pip install mutagen librosa numpy scipy")


def extract_audio_metadata(filepath: str, original_filename: str = None, original_timestamps: dict = None) -> dict:
    """Comprehensive Audio Metadata Extractor - Uncover every detail about your audio files"""
    
    def format_duration(seconds):
        """Enhanced duration formatting with context"""
        if not seconds or seconds <= 0:
            return "Unknown Duration"
        
        seconds = int(float(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        duration_str = ""
        if hours > 0:
            duration_str = f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            duration_str = f"{minutes}m {secs}s"
        else:
            duration_str = f"{secs}s"
        
        # Add context based on duration
        if seconds < 30:
            return f"{duration_str} (Very Short - Quick Message)"
        elif seconds < 300:  # 5 minutes
            return f"{duration_str} (Short - Voice Note/Song)"
        elif seconds < 1800:  # 30 minutes
            return f"{duration_str} (Medium Length - Typical Content)"
        elif seconds < 3600:  # 1 hour
            return f"{duration_str} (Long Form - Extended Content)"
        else:
            return f"{duration_str} (Very Long - Podcast/Audiobook Length)"

    def format_bitrate(bitrate):
        """Enhanced bitrate formatting with quality assessment"""
        if not bitrate:
            return "Unknown Quality"
        
        kbps = bitrate // 1000 if bitrate >= 1000 else bitrate
        
        if bitrate >= 320000:
            return f"{kbps} kbps (Excellent - Studio Quality)"
        elif bitrate >= 256000:
            return f"{kbps} kbps (Very High - Near CD Quality)"
        elif bitrate >= 192000:
            return f"{kbps} kbps (High - Good for Music)"
        elif bitrate >= 128000:
            return f"{kbps} kbps (Standard - Acceptable Quality)"
        elif bitrate >= 96000:
            return f"{kbps} kbps (Medium - Compressed)"
        elif bitrate >= 64000:
            return f"{kbps} kbps (Low - Voice Optimized)"
        elif bitrate >= 32000:
            return f"{kbps} kbps (Very Low - Basic Voice)"
        else:
            return f"{bitrate} bps (Extremely Low - Legacy Format)"

    def format_filesize(size_bytes):
        """Enhanced file size formatting with storage context"""
        if not size_bytes:
            return "Unknown Size"
        
        original_size = size_bytes
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                size_str = f"{size_bytes:.1f} {unit}" if unit != 'B' else f"{int(size_bytes)} {unit}"
                
                # Add storage context
                if original_size < 1024:
                    context = "(Tiny - Text Message Size)"
                elif original_size < 1024 * 1024:  # < 1MB
                    context = "(Small - Quick Transfer)"
                elif original_size < 10 * 1024 * 1024:  # < 10MB
                    context = "(Medium - Standard Audio)"
                elif original_size < 100 * 1024 * 1024:  # < 100MB
                    context = "(Large - High Quality Audio)"
                else:
                    context = "(Very Large - Uncompressed/Long Content)"
                
                return f"{size_str} {context}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f} TB (Massive File)"

    def format_timestamp(timestamp):
        """Enhanced timestamp formatting with age context"""
        try:
            if timestamp:
                dt = datetime.fromtimestamp(timestamp)
                age_days = (datetime.now() - dt).days
                formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                
                if age_days == 0:
                    return f"{formatted_date} (Today - Fresh)"
                elif age_days == 1:
                    return f"{formatted_date} (Yesterday)"
                elif age_days <= 7:
                    return f"{formatted_date} ({age_days} days ago - Recent)"
                elif age_days <= 30:
                    return f"{formatted_date} ({age_days // 7} weeks ago)"
                elif age_days <= 365:
                    return f"{formatted_date} ({age_days // 30} months ago)"
                elif age_days <= 365 * 5:
                    return f"{formatted_date} ({age_days // 365} years ago)"
                else:
                    return f"{formatted_date} ({age_days // 365} years ago - Vintage)"
        except:
            pass
        return "Unknown Date"

    def detect_audio_type_advanced(filepath, duration_seconds, bitrate, channels, sample_rate, file_size):
        """Advanced audio type detection with multiple heuristics"""
        filename = Path(filepath).name.lower()
        
        # Pattern-based detection
        patterns = {
            'WhatsApp Voice Message': ['whatsapp', 'aud-', 'ptt-', 'wa-', 'voice-note'],
            'Voice Recording': ['recording', 'voice', 'memo', 'rec_', 'dictation'],
            'Phone Call Recording': ['call', 'phone', 'conversation'],
            'Music Track': ['music', 'song', 'track', 'album', 'artist'],
            'Podcast Episode': ['podcast', 'episode', 'show', 'interview'],
            'Audiobook': ['audiobook', 'book', 'chapter', 'narration'],
            'Radio Show': ['radio', 'fm', 'am', 'broadcast'],
            'Conference Call': ['meeting', 'conference', 'zoom', 'teams'],
            'Voicemail': ['voicemail', 'message', 'vm']
        }
        
        for audio_type, keywords in patterns.items():
            if any(keyword in filename for keyword in keywords):
                return audio_type
        
        # Heuristic-based detection
        if duration_seconds and bitrate and channels:
            # Very short, low bitrate, mono = voice message
            if duration_seconds < 60 and bitrate < 64000 and channels == 1:
                return "Voice Message (Quick Note)"
            
            # Long duration, good quality, stereo = music/podcast
            elif duration_seconds > 180 and bitrate >= 128000 and channels == 2:
                if duration_seconds > 1800:  # > 30 minutes
                    return "Long-form Audio (Podcast/Audiobook/Mix)"
                else:
                    return "Music Track (Standard Length)"
            
            # Mono, speech-optimized bitrate
            elif channels == 1 and 32000 <= bitrate <= 96000:
                return "Speech Recording (Optimized for Voice)"
            
            # High quality, short duration
            elif bitrate >= 192000 and duration_seconds < 600:  # < 10 minutes
                return "High Quality Audio (Studio/Demo)"
            
            # Very long, any quality
            elif duration_seconds > 3600:  # > 1 hour
                return "Extended Content (Lecture/Audiobook/Mix)"
        
        # File size heuristics
        if file_size and duration_seconds:
            bytes_per_second = file_size / duration_seconds
            if bytes_per_second > 20000:  # High data rate
                return "High Fidelity Audio (Uncompressed/Lossless)"
            elif bytes_per_second < 8000:  # Low data rate
                return "Highly Compressed Audio (Bandwidth Optimized)"
        
        return "General Audio File"

    def calculate_file_fingerprint(filepath):
        """Generate multiple fingerprints for the audio file"""
        fingerprints = {}
        
        try:
            # MD5 hash of entire file
            with open(filepath, 'rb') as f:
                md5_hash = hashlib.md5()
                sha256_hash = hashlib.sha256()
                
                chunk_size = 8192
                file_chunks = 0
                
                while chunk := f.read(chunk_size):
                    md5_hash.update(chunk)
                    sha256_hash.update(chunk)
                    file_chunks += 1
                
                fingerprints['MD5 Hash'] = md5_hash.hexdigest()
                fingerprints['SHA256 Hash'] = sha256_hash.hexdigest()[:32] + "..."
                fingerprints['File Chunks Processed'] = f"{file_chunks} chunks of {chunk_size} bytes"
        except Exception as e:
            fingerprints['Hash Error'] = f"Could not generate hash: {str(e)}"
        
        return fingerprints

    def analyze_audio_properties_advanced(filepath):
        """Advanced audio analysis using librosa"""
        try:
            # Load audio for analysis (first 60 seconds to save time)
            y, sr = librosa.load(filepath, duration=60, sr=None)
            
            analysis = {}
            
            # Loudness analysis
            rms = librosa.feature.rms(y=y)[0]
            rms_db = librosa.amplitude_to_db(rms)
            
            analysis['RMS Energy (dB)'] = f"{np.mean(rms_db):.2f} dB (Average Loudness)"
            analysis['Dynamic Range'] = f"{np.max(rms_db) - np.min(rms_db):.2f} dB"
            
            # Spectral analysis
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            analysis['Spectral Centroid'] = f"{np.mean(spectral_centroids):.0f} Hz (Brightness)"
            
            # Zero crossing rate (speech vs music indicator)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            avg_zcr = np.mean(zcr)
            
            if avg_zcr > 0.1:
                zcr_type = "High (Likely Speech/Noisy)"
            elif avg_zcr > 0.05:
                zcr_type = "Medium (Mixed Content)"
            else:
                zcr_type = "Low (Likely Music/Tonal)"
            
            analysis['Zero Crossing Rate'] = f"{avg_zcr:.4f} ({zcr_type})"
            
            # Tempo estimation (for music)
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                analysis['Estimated Tempo'] = f"{tempo:.0f} BPM"
                analysis['Beat Confidence'] = "High" if len(beats) > 10 else "Low/No Beat Detected"
            except:
                analysis['Estimated Tempo'] = "Could not detect tempo"
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            analysis['Spectral Rolloff'] = f"{np.mean(rolloff):.0f} Hz (Energy Distribution)"
            
            # MFCC analysis (voice characteristics)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            analysis['MFCC Variance'] = f"{np.var(mfccs):.4f} (Voice Pattern Complexity)"
            
            return analysis
            
        except Exception as e:
            return {'Analysis Error': f"Could not analyze audio properties: {str(e)}"}

    def detect_file_format_details(filepath):
        """Detailed file format analysis"""
        details = {}
        
        try:
            with open(filepath, 'rb') as f:
                header = f.read(100)
                
                # Detailed format detection
                if header.startswith(b'ID3'):
                    details['Format Type'] = "MP3 with ID3 tags"
                    version = header[3]
                    details['ID3 Version'] = f"ID3v2.{version}"
                elif header.startswith(b'RIFF'):
                    details['Format Type'] = "WAV (Uncompressed PCM)"
                    if b'WAVE' in header[:20]:
                        details['Container'] = "WAVE container"
                elif header.startswith(b'\xff\xfb') or header.startswith(b'\xff\xf3'):
                    details['Format Type'] = "MP3 (MPEG Audio)"
                    details['MPEG Layer'] = "Layer III"
                elif header.startswith(b'fLaC'):
                    details['Format Type'] = "FLAC (Free Lossless Audio Codec)"
                    details['Compression'] = "Lossless"
                elif header.startswith(b'OggS'):
                    details['Format Type'] = "OGG container"
                    if b'vorbis' in header.lower():
                        details['Codec'] = "Vorbis"
                elif header.startswith(b'ftypM4A'):
                    details['Format Type'] = "M4A (MPEG-4 Audio)"
                    details['Container'] = "MP4"
                else:
                    details['Format Type'] = "Unknown or custom format"
                
                # File signature analysis
                details['File Signature'] = ' '.join(f'{b:02X}' for b in header[:16])
                
        except Exception as e:
            details['Format Analysis Error'] = str(e)
        
        return details

    def extract_comprehensive_tags(audio_file):
        """Extract and categorize all possible tags"""
        if not audio_file.tags:
            return {"Tag Status": "No metadata tags found"}
        
        categorized_tags = {
            "Basic Information": {},
            "Album & Release Info": {},
            "Technical Details": {},
            "Extended Metadata": {},
            "Custom Tags": {}
        }
        
        # Tag mapping for better organization
        tag_categories = {
            'Basic Information': ['TIT2', 'TPE1', 'TPE2', 'TCON', 'TDRC', 'title', 'artist', 'album', 'date', 'genre'],
            'Album & Release Info': ['TALB', 'TRCK', 'TPOS', 'TPE2', 'TPUB', 'albumartist', 'tracknumber', 'discnumber'],
            'Technical Details': ['TLEN', 'TBPM', 'TKEY', 'length', 'bpm', 'initialkey'],
            'Extended Metadata': ['COMM', 'TCOM', 'TPE3', 'TCOP', 'TENC', 'comment', 'composer', 'copyright', 'encodedby']
        }
        
        # Standard tag name mappings
        friendly_names = {
            'TIT2': 'Title', 'TPE1': 'Artist', 'TALB': 'Album', 'TDRC': 'Year',
            'TCON': 'Genre', 'TPE2': 'Album Artist', 'TRCK': 'Track Number',
            'TPOS': 'Disc Number', 'TPUB': 'Publisher', 'TLEN': 'Length',
            'TBPM': 'BPM', 'TKEY': 'Key', 'COMM': 'Comment', 'TCOM': 'Composer',
            'TPE3': 'Conductor', 'TCOP': 'Copyright', 'TENC': 'Encoded By',
            'title': 'Title', 'artist': 'Artist', 'album': 'Album', 'date': 'Date',
            'genre': 'Genre', 'albumartist': 'Album Artist', 'tracknumber': 'Track Number',
            'discnumber': 'Disc Number', 'comment': 'Comment', 'composer': 'Composer',
            'copyright': 'Copyright', 'encodedby': 'Encoded By', 'bpm': 'BPM'
        }
        
        # Process all tags
        for key, value in audio_file.tags.items():
            if not value:
                continue
            
            # Clean up value
            if isinstance(value, list):
                clean_value = ', '.join(str(v).strip() for v in value if str(v).strip())
            else:
                clean_value = str(value).strip()
            
            if not clean_value:
                continue
            
            # Get friendly name
            friendly_key = friendly_names.get(key, key.replace('©', '').replace('\xa9', '').title())
            
            # Categorize the tag
            categorized = False
            for category, tag_list in tag_categories.items():
                if key.lower() in [t.lower() for t in tag_list]:
                    categorized_tags[category][friendly_key] = clean_value
                    categorized = True
                    break
            
            # If not categorized, put in custom tags
            if not categorized:
                categorized_tags["Custom Tags"][friendly_key] = clean_value
        
        # Remove empty categories
        return {k: v for k, v in categorized_tags.items() if v}

    try:
        # File system investigation
        file_path = Path(filepath)
        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {filepath}",
                "metadata": None
            }
        
        stat = file_path.stat()
        file_size = stat.st_size
        
        # Use ORIGINAL timestamps if provided, otherwise fall back to temp file timestamps
        if original_timestamps:
            created_time = original_timestamps.get('created') or original_timestamps.get('modified')
            modified_time = original_timestamps.get('modified')
            access_time = original_timestamps.get('modified')  # Use modified as access time fallback
        else:
            created_time = stat.st_ctime
            modified_time = stat.st_mtime
            access_time = stat.st_atime
        
        # Use original filename for display if provided
        display_name = original_filename if original_filename else file_path.name
        display_path = Path(display_name)
        
        # Audio file analysis
        audio_file = File(filepath)
        if audio_file is None:
            return {
                "success": False,
                "error": f"Unsupported or corrupted audio file: {display_name}",
                "metadata": None
            }
        
        # Extract basic properties
        info = audio_file.info
        duration_seconds = getattr(info, 'length', 0)
        bitrate = getattr(info, 'bitrate', 0)
        sample_rate = getattr(info, 'sample_rate', None)
        channels = getattr(info, 'channels', None)
        bits_per_sample = getattr(info, 'bits_per_sample', None)
        
        # Build comprehensive metadata
        metadata = {
            "File System Information": {
                "File Name": display_name,
                "Full Path": display_name if original_filename else str(file_path.absolute()),
                "File Extension": display_path.suffix.upper().replace('.', '') or 'No Extension',
                "File Size": format_filesize(file_size),
                "Raw Size (Bytes)": f"{file_size:,} bytes",
                "Created": format_timestamp(created_time),
                "Last Modified": format_timestamp(modified_time),
                "Last Accessed": format_timestamp(access_time),
                "File Permissions": oct(stat.st_mode)[-3:] if not original_filename else "Upload Context",
            },
            
            "Audio Properties": {
                "Duration": format_duration(duration_seconds),
                "Precise Duration": f"{duration_seconds:.3f} seconds" if duration_seconds else "Unknown",
                "Audio Type": detect_audio_type_advanced(display_name, duration_seconds, bitrate, channels, sample_rate, file_size),
                "Bitrate": format_bitrate(bitrate),
                "Sample Rate": f"{sample_rate:,} Hz" if sample_rate else "Unknown",
                "Channels": f"{channels} ({'Mono' if channels == 1 else 'Stereo' if channels == 2 else f'{channels}-Channel Surround'})" if channels else "Unknown",
                "Bit Depth": f"{bits_per_sample}-bit" if bits_per_sample else "Unknown",
                "Audio Format": getattr(info, 'mime', ['Unknown'])[0] if hasattr(info, 'mime') else 'Unknown Format',
            },
            
            "Format Analysis": detect_file_format_details(filepath),
            "File Fingerprints": calculate_file_fingerprint(filepath),
            "Advanced Audio Analysis": analyze_audio_properties_advanced(filepath),
            "Metadata Tags": extract_comprehensive_tags(audio_file),
        }
        
        # Add codec-specific technical details
        codec_details = {}
        technical_attributes = ['version', 'layer', 'mode', 'protected', 'sketchy', 'encoder_info', 'encoder_settings']
        
        for attr in technical_attributes:
            if hasattr(info, attr):
                value = getattr(info, attr)
                if value is not None:
                    codec_details[attr.replace('_', ' ').title()] = str(value)
        
        if codec_details:
            metadata["Codec Technical Details"] = codec_details
        
        # Calculate additional statistics
        if duration_seconds and bitrate and file_size:
            estimated_size = (bitrate * duration_seconds) / 8  # Convert bits to bytes
            compression_ratio = estimated_size / file_size if file_size > 0 else 0
            
            metadata["Calculated Statistics"] = {
                "Estimated Uncompressed Size": format_filesize(estimated_size),
                "Compression Efficiency": f"{(1 - compression_ratio) * 100:.1f}%" if compression_ratio <= 1 else "File larger than expected",
                "Data Rate": f"{file_size / duration_seconds:.0f} bytes/second" if duration_seconds > 0 else "Unknown",
                "Bits Per Second": f"{bitrate:,} bps" if bitrate else "Unknown",
            }
        
        return {
            "success": True,
            "error": None,
            "metadata": metadata
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error analyzing file: {str(e)}",
            "metadata": None
        }