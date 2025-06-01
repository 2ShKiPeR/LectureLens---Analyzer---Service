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
    count = dict(Counter(phrases))
    sorted_result = dict(sorted(count.items(), key=lambda x: -x[1]))
    return {"keywords": sorted_result}
