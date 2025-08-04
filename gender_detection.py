import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Load model and extractor only once
model_name = "prithivMLmods/Common-Voice-Gender-Detection"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
id2label = {"0": "Female", "1": "Male"}

def process_audio(path: str) -> dict:
    try:
        speech, sr = librosa.load(path, sr=16000)
        inputs = feature_extractor(speech, sampling_rate=sr, padding=True, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
            predicted_label = torch.argmax(probs).item()

        return {"gender": id2label[str(predicted_label)]}

    except Exception as e:
        return {"error": str(e)}
