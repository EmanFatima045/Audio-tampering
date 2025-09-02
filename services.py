from sqlalchemy.orm import Session
from models import AudioForensicsCase
from datetime import datetime
import os
import uuid

class AudioForensicsService:
    def __init__(self, db: Session):
        self.db = db
    
    def create_case(self, name: str, original_filename: str, file_content: bytes, notes: str = None) -> AudioForensicsCase:
        """Create a new case and save the audio file"""
        # Generate unique case ID
        case_id = str(uuid.uuid4())
        
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads/cases"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save file with case ID
        file_extension = os.path.splitext(original_filename)[1]
        file_path = os.path.join(uploads_dir, f"{case_id}{file_extension}")
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Create case record
        case = AudioForensicsCase(
            id=case_id,
            case_name=name,
            original_filename=original_filename,
            file_path=file_path,
            notes=notes
        )
        
        self.db.add(case)
        self.db.commit()
        self.db.refresh(case)
        
        return case
    
    def get_case(self, case_id: str) -> AudioForensicsCase:
        """Get case by ID"""
        return self.db.query(AudioForensicsCase).filter(AudioForensicsCase.id == case_id).first()
    
    def get_all_cases(self) -> list[AudioForensicsCase]:
        """Get all cases"""
        return self.db.query(AudioForensicsCase).order_by(AudioForensicsCase.created_at.desc()).all()
    
    def delete_case(self, case_id: str) -> bool:
        """Delete case and associated file"""
        case = self.get_case(case_id)
        if not case:
            return False
        
        # Delete file
        if os.path.exists(case.file_path):
            os.remove(case.file_path)
        
        # Delete case
        self.db.delete(case)
        self.db.commit()
        return True
    
    def update_transcription(self, case_id: str, text: str, confidence: float = None, language: str = None):
        """Update transcription results"""
        case = self.get_case(case_id)
        if case:
            case.transcription_text = text
            case.transcription_confidence = confidence
            case.transcription_language = language
            case.transcription_completed = "completed"
            self.db.commit()
            self.db.refresh(case)
        return case
    
    def update_sentiment(self, case_id: str, sentiment: str, confidence: float = None):
        """Update sentiment analysis results"""
        case = self.get_case(case_id)
        if case:
            case.sentiment_result = sentiment
            case.sentiment_confidence = confidence
            case.sentiment_completed = "completed"
            self.db.commit()
            self.db.refresh(case)
        return case
    
    def update_gender_detection(self, case_id: str, gender: str, confidence: float = None):
        """Update gender detection results"""
        case = self.get_case(case_id)
        if case:
            case.gender_result = gender
            case.gender_confidence = confidence
            case.gender_completed = "completed"
            self.db.commit()
            self.db.refresh(case)
        return case
    
    def update_metadata(self, case_id: str, metadata_json: dict, original_timestamps: dict = None):
        """Update metadata analysis results"""
        case = self.get_case(case_id)
        if case:
            case.metadata_json = metadata_json
            case.original_timestamps = original_timestamps
            case.metadata_completed = "completed"
            self.db.commit()
            self.db.refresh(case)
        return case
    
    def update_temporal_analysis(self, case_id: str, background_splices: list, phase_splices: list, combined_splices: list):
        """Update temporal inconsistency analysis results (no graph)"""
        case = self.get_case(case_id)
        if case:
            case.background_splices = background_splices
            case.phase_splices = phase_splices
            case.combined_splices = combined_splices
            case.temporal_completed = "completed"
            self.db.commit()
            self.db.refresh(case)
        return case
    
    def update_diarization(self, case_id: str, estimated_speakers: int, segments: list):
        """Update diarization results with segments"""
        case = self.get_case(case_id)
        if case:
            case.estimated_speakers = estimated_speakers
            case.diarization_segments = segments
            case.diarization_completed = "completed"
            self.db.commit()
            self.db.refresh(case)
        return case
    
    def update_segment_analysis(self, case_id: str, segment_index: int, transcription: str = None, 
                               sentiment: str = None, gender: str = None):
        """Update individual segment analysis results"""
        case = self.get_case(case_id)
        if case and case.diarization_segments:
            segments = case.diarization_segments.copy()
            if segment_index < len(segments):
                if transcription is not None:
                    segments[segment_index]['transcription'] = transcription
                if sentiment is not None:
                    segments[segment_index]['sentiment'] = sentiment
                if gender is not None:
                    segments[segment_index]['gender'] = gender
                
                case.diarization_segments = segments
                self.db.commit()
                self.db.refresh(case)
        return case
