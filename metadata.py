
import os
from pathlib import Path
from datetime import datetime

try:
    from mutagen import File
except ImportError:
    raise ImportError("mutagen library required: pip install mutagen")


def extract_audio_metadata(filepath: str) -> dict:
   
    def format_duration(seconds):
        """Convert seconds to human readable format"""
        if not seconds or seconds <= 0:
            return "Unknown"
        
        seconds = int(float(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def format_bitrate(bitrate):
        """Format bitrate"""
        if not bitrate:
            return "Unknown"
        return f"{bitrate} bps" if bitrate < 1000 else f"{bitrate // 1000} kbps"

    def format_filesize(size_bytes):
        """Convert bytes to human readable format"""
        if not size_bytes:
            return "Unknown"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def format_timestamp(timestamp):
        """Format timestamp to readable date"""
        try:
            if timestamp:
                return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        except:
            pass
        return "Unknown"

    def detect_audio_type(filepath, duration_seconds, bitrate):
        """Detect the likely type of audio file"""
        filename = Path(filepath).name.lower()
        
        # Check filename patterns
        if 'whatsapp' in filename or 'aud-' in filename or 'ptt-' in filename:
            return "WhatsApp Voice Message"
        elif 'recording' in filename or 'voice' in filename:
            return "Voice Recording"
        elif 'call' in filename:
            return "Call Recording"
        elif filename.startswith('rec_'):
            return "Audio Recording"
        
        # Check duration and quality for hints
        if duration_seconds and duration_seconds < 300:  # Less than 5 minutes
            if bitrate and bitrate < 64000:  # Low bitrate
                return "Voice Message/Recording"
        
        return "Audio File"

    try:
        # Validate file exists
        file_path = Path(filepath)
        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {filepath}",
                "metadata": None
            }
        
        # Get file system info
        stat = file_path.stat()
        file_size = stat.st_size
        created_time = stat.st_ctime
        modified_time = stat.st_mtime
        
        # Try to load audio file
        audio_file = File(filepath)
        if audio_file is None:
            return {
                "success": False,
                "error": f"Unsupported or corrupted audio file: {filepath}",
                "metadata": None
            }
        
        # Extract basic audio properties
        info = audio_file.info
        duration_seconds = getattr(info, 'length', 0)
        bitrate = getattr(info, 'bitrate', 0)
        sample_rate = getattr(info, 'sample_rate', None)
        channels = getattr(info, 'channels', None)
        
        # Build metadata structure
        metadata = {
            "file_info": {
                "filename": file_path.name,
                "file_path": str(file_path.absolute()),
                "file_size_bytes": file_size,
                "file_size_formatted": format_filesize(file_size),
                "file_extension": file_path.suffix.upper().replace('.', ''),
                "created_timestamp": created_time,
                "created_formatted": format_timestamp(created_time),
                "modified_timestamp": modified_time,
                "modified_formatted": format_timestamp(modified_time)
            },
            "audio_properties": {
                "duration_seconds": duration_seconds,
                "duration_formatted": format_duration(duration_seconds),
                "format": getattr(info, 'mime', ['Unknown'])[0] if hasattr(info, 'mime') else 'Unknown',
                "bitrate": bitrate,
                "bitrate_formatted": format_bitrate(bitrate),
                "sample_rate": sample_rate,
                "sample_rate_formatted": f"{sample_rate} Hz" if sample_rate else 'Unknown',
                "channels": channels,
                "channels_formatted": f"{channels} ({'Mono' if channels == 1 else 'Stereo' if channels == 2 else 'Multi-channel'})" if channels else 'Unknown',
                "bits_per_sample": getattr(info, 'bits_per_sample', None),
                "estimated_type": detect_audio_type(filepath, duration_seconds, bitrate)
            },
            "tags": {},
            "codec_details": {}
        }
        
        # Extract any available tags/metadata
        if audio_file.tags:
            for key, value in audio_file.tags.items():
                if value:
                    # Clean up the value
                    if isinstance(value, list):
                        clean_value = ', '.join(str(v) for v in value if v)
                    else:
                        clean_value = str(value).strip()
                    
                    if clean_value:
                        # Clean up common tag names
                        clean_key = key.replace('©', '').replace('\xa9', '').upper()
                        metadata["tags"][clean_key] = clean_value
        
        # Add codec-specific information
        codec_info = {}
        if hasattr(info, 'version'):
            codec_info['version'] = str(info.version)
        if hasattr(info, 'layer'):
            codec_info['layer'] = str(info.layer)
        if hasattr(info, 'mode'):
            codec_info['mode'] = str(info.mode)
        if hasattr(info, 'protected'):
            codec_info['protected'] = str(info.protected)
        if hasattr(info, 'sketchy'):
            codec_info['sketchy'] = str(info.sketchy)
        
        metadata["codec_details"] = codec_info
        
        return {
            "success": True,
            "error": None,
            "metadata": metadata
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading file: {str(e)}",
            "metadata": None
        }


