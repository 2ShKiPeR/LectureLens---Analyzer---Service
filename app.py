from fastapi import FastAPI, File, UploadFile
import whisper
import tempfile
import shutil
from sentiment import router as sentiment_router
from keywords import router as keywords_router

app = FastAPI()
app.include_router(sentiment_router)
app.include_router(keywords_router)

model = whisper.load_model("base")  # можешь использовать "small" или "tiny" для ускорения

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        shutil.copyfileobj(audio.file, temp_audio)
        temp_audio_path = temp_audio.name

    result = model.transcribe(temp_audio_path)
    return {"text": result["text"]}

