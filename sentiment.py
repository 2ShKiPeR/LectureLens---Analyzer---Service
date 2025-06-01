from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

router = APIRouter()

model_id = "blanchefort/rubert-base-cased-sentiment"
tokenizer = None
model = None
labels = ['negative', 'neutral', 'positive']

class TextRequest(BaseModel):
    text: str

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("🔁 Загружаю модель для анализа тональности...")
        tokenizer_local = AutoTokenizer.from_pretrained(model_id)
        model_local = AutoModelForSequenceClassification.from_pretrained(model_id)
        globals()["tokenizer"] = tokenizer_local
        globals()["model"] = model_local
        print("✅ Модель загружена!")

@router.post("/analyze-sentiment")
def analyze_sentiment(req: TextRequest):
    load_model()

    inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[0].numpy()
        label_id = scores.argmax()

    return {
        "sentiment": labels[label_id],
        "score_vector": scores.tolist()
    }
