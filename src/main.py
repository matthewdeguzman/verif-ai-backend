import shutil
import os

from moviepy.editor import VideoFileClip
from fastapi import FastAPI, UploadFile, HTTPException
from pysbd.utils import PySBDFactory
import spacy
from pydantic import BaseModel
from dotenv import dotenv_values
import requests

from pytubefix import YouTube
from pytubefix.cli import on_progress

app = FastAPI()


def extract_audio_from_mp4(mp4_file_path, output_audio_path):
    # Load the video file
    video_clip = VideoFileClip(mp4_file_path)

    # Extract the audio
    audio_clip = video_clip.audio

    # Write the audio file to the specified output path
    audio_clip.write_audiofile(output_audio_path)

    # Close the video and audio clips to release resources
    video_clip.close()
    audio_clip.close()


def download_audio(url: str, output_path: str, filename: str):
    """Download audio from url."""
    yt = YouTube(url, on_progress_callback=on_progress)
    print(yt.title)
    ys = yt.streams.get_highest_resolution()
    ys.download(output_path=output_path, filename=filename)

    # Example usage
    mp4_file_path = "video.mp4"
    output_audio_path = "audio.wav"  # You can also use .wav or other supported formats

    extract_audio_from_mp4(mp4_file_path, output_audio_path)


def parse_audio(audio_path: str) -> list[str]:
    """Parse audio from file in `audio_path using OpenAI Whisper`"""
    config = dotenv_values(".env")
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
    headers = {"Authorization": f"Bearer {config['HF_TOKEN']}"}

    def query(filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)
        return response.json()

    output = query(audio_path)
    print(output)
    text = output["text"]
    nlp = spacy.load("en_core_web_sm")
    return [str(i) for i in nlp(text).sents]


class TranscribeResult(BaseModel):
    status: int
    message: str
    text: list[str]


# class Speech(BaseModel):
#     text: list[str]

# class Decision(BaseModel):
#     facts: list[str]
#     lies: list[str]

# class FactCheckResult(BaseModel):
#     status: int
#     message: str
#     results: Decision


def _transcribe(audio_path: str, output_dir: str) -> TranscribeResult:
    text = parse_audio(audio_path)

    # Clean up
    shutil.rmtree(output_dir)

    return TranscribeResult(status=200, message="success", text=text)


@app.post("/transcribe_url")
def transcribe_url(video_url: str) -> TranscribeResult:
    output_dir = "./downloaded_media"
    video_name = "video.mp4"
    audio_path = os.path.join(output_dir, video_name)
    download_audio(video_url, output_path=output_dir, filename=video_name)

    return _transcribe(audio_path=audio_path, output_dir=output_dir)


@app.post("/transcribe_file")
async def transcribe_file(file: UploadFile) -> TranscribeResult:
    output_dir = "./downloaded_media"
    audio_path = os.path.join(output_dir, "audio.wav")
    video_path = os.path.join(output_dir, "video.mp4")

    if file.filename[-4:] != ".mp4":
        raise HTTPException(
            status_code=400, detail=f"Invalid file extension: '{file.filename[:-4]}'. Only .mp4 files are allowed."
        )

    contents = await file.read()
    with open(video_path, "wb") as f:
        f.write(contents)

    extract_audio_from_mp4(mp4_file_path=video_path, output_audio_path=audio_path)
    return _transcribe(audio_path=audio_path, output_dir=output_dir)


# @app.post("/fact_check")
# def fact_check(speech: Speech) -> FactCheckResult:
#     return FactCheckResult(status=200, message="success", results=Decision(facts=["hello"], lies=["world"]))
