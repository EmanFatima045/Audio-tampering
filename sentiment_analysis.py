from transformers import pipeline

# Load sentiment pipeline once
classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def get_sentiment(text: str) -> str:
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]

    if label == "POSITIVE" and score > 0.7:
        return "Positive"
    elif label == "NEGATIVE" and score > 0.7:
        return "Negative"
    else:
        return "Neutral"
