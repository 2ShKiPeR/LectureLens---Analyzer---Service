from fastapi import APIRouter
from pydantic import BaseModel
from multi_rake import Rake
from collections import Counter

router = APIRouter()
rake = Rake()

class TextRequest(BaseModel):
    text: str

@router.post("/extract-keywords")
def extract_keywords(req: TextRequest):
    keywords = rake.apply(req.text)
    phrases = [kw[0] for kw in keywords]
    count = Counter(phrases)

    return [{"word": k, "frequency": v} for k, v in count.items()]