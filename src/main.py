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

app = FastAPI()
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


def scrape_additional_sources(claim):
    return "hi"

def get_descriptions(claim):
    search_url = f"https://www.google.com/search?q={claim.replace(' ', '+')}"
    options = Options()
    options.add_argument('--headless')  # Run in headless mode
    
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
    soup = BeautifulSoup(rendered_html, 'html.parser')

    # for description in soup.find_all("a", {"jsname":"UWckNb"}, limit=5):
    descriptions = []

    for description in soup.find_all("div", {"class":"VwiC3b yXK7lf lVm3ye r025kc hJNv6b Hdw6tb"}, limit=5):
        text = description.get_text()
        descriptions.append(text)

    return descriptions

def get_source_links(claim):
    search_url = f"https://www.google.com/search?q={claim.replace(' ', '+')}"
    options = Options()
    options.add_argument('--headless')  # Run in headless mode
    
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
    soup = BeautifulSoup(rendered_html, 'html.parser')

    links = []
    for link in soup.find_all("a", {"jsname":"UWckNb"}, limit=5):
        url = link['href']
        links.append(url)

    return links

def retrieve_sources(claim: str):
    # Query the Google Fact Check Tools API
    fact_check_results = query_fact_check_api(claim)
    # print(json.dumps(fact_check_results, indent=2))
    supported_publishers = ["full fact", "africa check", "snopes", "the new york times", "politifact", "usa today"]
    reliable_claims = []

    if "claims" in fact_check_results:
        for result in fact_check_results["claims"]:
            # print(f"Claim: {result['text']}")
            # print(f"Claimant: {result.get('claimant')}")
            # print(f"Claim Date: {result.get('claimDate')}")
            for review in result["claimReview"]:
                if review["publisher"]["name"].lower() in supported_publishers:
                    reliable_claims.append(result)
                # print(f"Publisher: {review['publisher']['name']}")
                # print(f"Title: {review['title']}")
                # print(f"URL: {review['url']}")
                # print(f"Rating: {review['textualRating']}")
                # print()
    else:
        scrape_additional_sources(claim)

    # print(f"Filtered sources:", json.dumps(reliable_claims, indent=2))
    return reliable_claims


def compare_claim_with_source(claim, source_text):
    doc1 = nlp(claim)
    doc2 = nlp(source_text)

    similarity = doc1.similarity(doc2)
    print(f"Similarity: {similarity}")

    return similarity > 0.5


def process_and_verify_claims(speech: Speech):

    claim_keywords = ["claim", "claims", "states", "reports", "says", "%"]
    claims = [segment for segment in speech.segments if any(keyword in segment for keyword in claim_keywords)]
    claim_results = {"verified": [], "unverified": [], "false": []}

    for claim in claims:
        print(f"Processing claim: {claim}")
        sources = retrieve_sources(claim)
        support, contradict, reviews = [], [], []
        for source in sources:
            for review in source["claimReview"]:
                if compare_claim_with_source(claim, review["textualRating"]):
                    support.append(review)
                else:
                    contradict.append(review)
                reviews.append(review)

        if (len(support) > 0 and len(contradict) > 0) or (len(support) == 0 and len(contradict) == 0):
            claim_results["unverified"].append({"claim": claim, "sources": reviews})
        elif len(support) > 0:
            claim_results["verified"].append({"claim": claim, "sources": reviews})
        else:
            claim_results["false"].append({"claim": claim, "sources": reviews})

    return claim_results


# results = process_and_verify_claims("i claim the world is not flat")
# print(json.dumps(results, indent=2))


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
    config = dotenv_values(".env")
    descriptions = (" ").join(get_descriptions(claim))
    client = OpenAI(api_key=config.get("OPENAI_API_KEY")) # for demo we can hardcode api key i think
   
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a fact-checking assistant. Verify the following claim based on the following descriptions from reliable sources and provide a response. Descriptions: (" + descriptions + ")"},
            {"role": "user", "content": claim}
        ]
    )

    verification = response.choices[0].message.content
    
    return verification


def parse_audio(audio_path: str) -> list[str]:
    """Parse audio from file in `audio_path using OpenAI Whisper`"""
    config = dotenv_values(".env")

    load_dotenv()
    with open("audio.mp3", "rb") as f:
        client = OpenAI(api_key=config.get("OPENAI_API_KEY"))
        transcript = client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language="en",
        )

    aggregated = ".".join([el["text"] for el in transcript.segments])
    return {"segments": transcript.segments, "text": aggregated}


@app.post("/transcribe_url")
def transcribe_url(video_url: str):
    output_dir = "./downloaded_media"
    video_name = "video.mp4"
    audio_path = os.path.join(output_dir, "audio.mp3")
    download_audio(video_url, output_path=output_dir, video_name=video_name, audio_name="audio.mp3")

    audio = parse_audio(audio_path)
    print(audio)

    # Clean up
    shutil.rmtree(output_dir)

    return TranscribedAudio(segments=audio["segments"], aggregated=audio["text"])


def main():
    claim = "Eating ice cream does not help remedy your cold"
    response = verify_claims_with_openai(claim)
    links = get_source_links(claim)
    print(response)
    print(links)

if __name__ == "__main__":
    main()

# @app.post("/fact_check")
# def fact_check(speech: Speech) -> FactCheckResult:
#     process_and_verify_claims(speech.text)
#     return FactCheckResult(status=200, message="success", results=Decision(facts=["hello"], lies=["world"]))

