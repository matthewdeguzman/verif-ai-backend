import shutil
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from fastapi.middleware.cors import CORSMiddleware
from moviepy.editor import VideoFileClip
from fastapi import FastAPI, UploadFile, HTTPException
from pysbd.utils import PySBDFactory
import spacy
from pydantic import BaseModel
from dotenv import dotenv_values
import requests
from transformers import pipeline
from bs4 import BeautifulSoup

from pytubefix import YouTube
from pytubefix.cli import on_progress

app = FastAPI()
nlp = spacy.load("en_core_web_sm")
origins = ["https://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def extract_claims(text: str):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    # print("Sentences: ", sentences)
    # print("Entities: ", entities)

    claim_keywords = ["claims", "states", "reports", "says"]
    claims = [sentence for sentence in sentences if any(keyword in sentence for keyword in claim_keywords)]
    # print("Claims:", claims)

    claim_contexts = {}

    for claim in claims:
        context_entities = [ent for ent in entities if ent[0] in claim]
        context_sentences = [
            sent for sent in sentences if sent != claim and any(ent[0] in sent for ent in context_entities)
        ]
        claim_contexts[claim] = {"entities": context_entities, "context_sentences": context_sentences}

    for claim, context in claim_contexts.items():
        print(f"Claim: {claim}")
        print(f"Entities: {context['entities']}")
        print(f"Context Sentences: {context['context_sentences']}")


def query_fact_check_api(claim):
    params = {"key": os.getenv("GOOGLE_API_KEY"), "query": claim, "languageCode": "en"}
    response = requests.get(GOOGLE_FACT_CHECK_URL, params=params)
    return response.json()


def scrape_additional_sources(claim):
    search_url = f"https://www.google.com/search?q={claim.replace(' ', '+')}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a"):
        href = link.get("href")
        if href and "url?q=" in href and not "webcache" in href:
            url = href.split("url?q=")[1].split("&")[0]
            print(f"Scraped URL: {url}")
            # Additional scraping and NLP can be done on the target page


def verify_claim(claim: str):
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
    # if similarity > 0.8:  # Threshold for considering a match
    #     print(f"The claim '{claim}' is likely true based on the source.")
    # else:
    #     print(f"The claim '{claim}' does not match well with the source.")


def process_and_verify_claims():
    doc = nlp(transcribed_text)

    sentences = [sent.text for sent in doc.sents]
    claim_keywords = ["claim", "claims", "states", "reports", "says"]
    claims = [sentence for sentence in sentences if any(keyword in sentence for keyword in claim_keywords)]
    claim_results = {"verified": [], "unverified": [], "false": []}

    for claim in claims:
        print(f"Processing claim: {claim}")
        results = verify_claim(claim)
        verified = False
        reviews = []
        for source in results:
            for review in source["claimReview"]:
                verified = max(compare_claim_with_source(claim, review["textualRating"]), verified)
                reviews.append(review)
        if verified:
            claim_results["verified"].append({"claim": claim, "sources": reviews})
        else:
            claim_results["unverified"].append({"claim": claim, "sources": reviews})

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


# @app.post("/fact_check")
# def fact_check(speech: Speech) -> FactCheckResult:
#     process_and_verify_claims(speech.text)
#     return FactCheckResult(status=200, message="success", results=Decision(facts=["hello"], lies=["world"]))
