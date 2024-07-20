from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()


class TranscribeResult(BaseModel):
    status: int
    message: str
    text: list[str]

class Speech(BaseModel):
    text: list[str]

class Decision(BaseModel):
    facts: list[str]
    lies: list[str]

class FactCheckResult(BaseModel):
    status: int
    message: str
    results: Decision

@app.post("/transcribe_url")
def transcribe_url(video_url: str) -> TranscribeResult:
    return TranscribeResult(status=200, message="success", text=["hello", "world"])

@app.post("/transcribe_file")
def transcribe_file(file: UploadFile) -> TranscribeResult:
    return TranscribeResult(status=200, message="success", text=["hello", "world"])

@app.post("/fact_check")
def fact_check(speech: Speech) -> FactCheckResult:
    return FactCheckResult(status=200, message="success", results=Decision(facts=["hello"], lies=["world"]))
