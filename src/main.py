import shutil
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from fastapi.middleware.cors import CORSMiddleware
from moviepy.editor import VideoFileClip
from fastapi import FastAPI, UploadFile, HTTPException
from pysbd.utils import PySBDFactory
from pydantic import BaseModel
from dotenv import dotenv_values
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import spacy

from pytubefix import YouTube
from pytubefix.cli import on_progress

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

load_dotenv()

app = FastAPI()
client = OpenAI()
nlp = spacy.load("en_core_web_sm")
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


class TranscribedAudio(BaseModel):
    segments: list[dict]
    aggregated: str


class Speech(BaseModel):
    text: list[dict]


class Decision(BaseModel):
    verified: list[dict]
    unverified: list[dict]
    false: list[dict]


class FactCheckResult(BaseModel):
    status: int
    message: str
    results: Decision


def query_fact_check_api(claim):
    params = {"key": os.getenv("GOOGLE_API_KEY"), "query": claim, "languageCode": "en"}
    response = requests.get(GOOGLE_FACT_CHECK_URL, params=params)
    return response.json()


def get_descriptions(claim):
    search_url = f"https://www.google.com/search?q={claim.replace(' ', '+')}"
    options = Options()
    options.add_argument("--headless")  # Run in headless mode

    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # Fetch the webpage
    driver.get(search_url)
    time.sleep(5)  # Wait for the dynamic content to load

    # Get the rendered HTML content
    rendered_html = driver.page_source

    # Close the browser
    driver.quit()

    # Parse with BeautifulSoup
    soup = BeautifulSoup(rendered_html, "html.parser")

    # for description in soup.find_all("a", {"jsname":"UWckNb"}, limit=5):
    descriptions = []

    for description in soup.find_all("div", {"class": "VwiC3b yXK7lf lVm3ye r025kc hJNv6b Hdw6tb"}, limit=5):
        text = description.get_text()
        descriptions.append(text)

    return descriptions


def get_source_links(claim):
    search_url = f"https://www.google.com/search?q={claim.replace(' ', '+')}"
    options = Options()
    options.add_argument("--headless")  # Run in headless mode

    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # Fetch the webpage
    driver.get(search_url)
    time.sleep(5)  # Wait for the dynamic content to load

    # Get the rendered HTML content
    rendered_html = driver.page_source

    # Close the browser
    driver.quit()

    # Parse with BeautifulSoup
    soup = BeautifulSoup(rendered_html, "html.parser")

    links = []
    for link in soup.find_all("a", {"jsname": "UWckNb"}, limit=5):
        url = link["href"]
        links.append(url)

    return links


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


def download_audio(url: str, output_path: str, video_name: str, audio_name: str):
    """Download audio from url."""
    yt = YouTube(url, on_progress_callback=on_progress)
    print(yt.title)
    ys = yt.streams.get_highest_resolution()
    ys.download(output_path=output_path, filename=video_name)

    extract_audio_from_mp4(os.path.join(output_path, video_name), os.path.join(output_path, audio_name))


# Verify claims using OpenAI
def verify_claims_with_openai(claim: str) -> str:
    descriptions = (" ").join(get_descriptions(claim))

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a fact-checking assistant. Verify the following claim based on the following descriptions from reliable sources and provide a response. The first word of the response should be True, False, or Uncertain followed by a comma, Descriptions: ("
                + descriptions
                + ")",
            },
            {"role": "user", "content": claim},
        ],
    )

    verification = response.choices[0].message.content

    return verification


def parse_audio(audio_path: str) -> list[str]:
    """Parse audio from file in `audio_path using OpenAI Whisper`"""
    config = dotenv_values(".env")

    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language="en",
        )

    aggregated = ".".join([el["text"] for el in transcript.segments])
    return {"segments": transcript.segments, "text": aggregated}


def is_claim(sentence: str):
    prompt = """I will give you a sentence, and I want you to tell me whether it is a claim or not a claim. I don't care if you think the sentence is false, just focus on if it is a sentence that someone is claiming to be a fact. If you are not sure, then say "False". If you are sure, then say "True". I do not want you to make things up. Here are some examples.

    Sentence: "I believe that water is unhealthy for you"
    Is Claim: True

    Sentence: "Good morning, how are you today?"
    Is Claim: False

    Sentence: "The cheese pizza is the worst option on the menu"
    Is Claim: False

    Sentence: "According to the Wall Street Journal, you said that elephants are the coolest animals on the planet"
    Is Claim: True

    Sentence: "17% of people in Florida were born without bones"
    Is Claim: True

    Sentence: "Lebron James is a white man"
    Is Claim: True

    Sentence: "The pope lives in Nevada"
    Is Claim: True

    Sentence: "Smoking a Cigar every day is very healthy for you"
    Is Claim True

    Sentence: "I love drinking wine, I think it's the best thing in the world"
    Is Claim: False

    Sentence: "{sentence}"
    Is Claim:"""

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt.format(sentence=sentence),
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    text = response.choices[0].text.strip()
    return text.lower() == "true"


def verify_claim(claim: str):

    print(f"Processing claim: {claim}")
    links = get_source_links(claim)
    results = verify_claims_with_openai(claim)

    claimType = results.split(",")[0]

    if claimType == "True":
        status = "verified"
        # claim_results["verified"].append({"claim": claim, "sources": links, "explanation": results})
    elif claimType == "False":
        status = "false"
    else:
        status = "uncertain"

    return status, links, results


def _fact_check(text: list[dict]):
    results = []
    for claim in text:
        if is_claim(claim["text"]):
            claim_type, links, result = verify_claim(claim["text"])
            results.append(
                {
                    "text": claim["text"],
                    "claimType": claim_type,
                    "sources": links,
                    "textualReason": result,
                    "timeStampStart": claim["start"],
                    "timeStampEnd": claim["end"],
                }
            )
        else:
            results.append(
                {
                    "text": claim["text"],
                    "claimType": "",
                    "sources": [],
                    "textualReason": "",
                    "timeStampStart": claim["start"],
                    "timeStampEnd": claim["end"],
                }
            )
    print(json.dumps(results, indent=2))
    return results


def transcribe_url(video_url: str):
    output_dir = "./downloaded_media"
    video_name = "video.mp4"
    audio_path = os.path.join(output_dir, "audio.mp3")
    download_audio(video_url, output_path=output_dir, video_name=video_name, audio_name="audio.mp3")

    audio = parse_audio(audio_path)
    print(audio)

    # Clean up
    shutil.rmtree(output_dir)

    return audio["segments"]


@app.post("/fact_check")
def fact_check(video_url: str) -> list[dict]:
    segments = transcribe_url(video_url)
    return _fact_check(segments)
