"""
Module 5 - passing an image into a prompt with the Responses API.

Sends a customer's warranty claim photo to a vision-capable model deployment
alongside a text instruction, and reads back a triage note. The same call runs
twice, at detail="high" and detail="low", to compare cost and answer quality.

Before running:
    pip install -r requirements.txt
"""

import base64
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

RESOURCE_ENDPOINT = os.getenv("FOUNDRY_RESOURCE_ENDPOINT")
API_KEY = os.getenv("FOUNDRY_API_KEY")
MODEL_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT_NAME")

IMAGE_PATH = "ripped_bag.jpg"

INSTRUCTIONS = (
    "You are a warranty triage assistant for Globomantics, a company that "
    "sells outdoor equipment. Given a photo submitted with a claim, describe "
    "the visible damage, say whether it reads as a manufacturing defect or as "
    "wear and tear, and recommend one of: approve, deny, send for inspection. "
    "If the photo does not show enough to decide, say what is missing instead "
    "of guessing. Answer in under 100 words."
)

CLAIM_TEXT = "Claim GLO-4417. The customer says this happened on the first trip."

# Set DEMO_PAUSE=0 in the environment to run straight through.
PAUSE = os.getenv("DEMO_PAUSE", "1") != "0"


def pause() -> None:
    """Hold one section of output on screen until Enter."""
    if PAUSE:
        input("  -- Enter to continue --")
        print()


def image_data_url(path: str) -> str:
    """Read a local image and wrap it as a base64 data URL."""
    if not os.path.exists(path):
        sys.exit(f"No image at {path}.")

    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    # The other option is a public HTTPS URL, which the service fetches
    # itself, so a local file or anything private is out. Media type has to
    # match the file: PNG, JPEG, GIF, or WEBP.
    return f"data:image/jpg;base64,{encoded}"


def triage(client: OpenAI, data_url: str, detail: str):
    """Send the photo and the claim text as one user message."""
    # `content` is a list of parts - text and image side by side, and a
    # message can hold several images. Responses names them `input_text` and
    # `input_image` with a string `image_url`; Chat Completions uses `text`
    # and `image_url` with an object.
    #
    # `detail` trades accuracy against tokens: "high" reads full resolution,
    # "low" uses a fixed small version.
    return client.responses.create(
        model=MODEL_DEPLOYMENT,
        instructions=INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": CLAIM_TEXT},
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": detail,
                    },
                ],
            }
        ],
        # Images eat input tokens, so leave room for the answer.
        max_output_tokens=600,
    )


def report(response) -> None:
    """Print the note, then the token cost of getting it."""
    # A truncated answer still returns 200
    if response.status == "incomplete":
        print(f"  [truncated: {response.incomplete_details.reason}]")

    print(response.output_text)
    print(
        f"\n  input tokens: {response.usage.input_tokens}"
        f"   output tokens: {response.usage.output_tokens}"
    )


def main() -> None:
    if not RESOURCE_ENDPOINT or not API_KEY or not MODEL_DEPLOYMENT:
        sys.exit(
            "Missing configuration. Copy .env.example to .env and fill in "
            "FOUNDRY_RESOURCE_ENDPOINT, FOUNDRY_API_KEY, and "
            "MODEL_DEPLOYMENT_NAME."
        )

    # The resource endpoint plus /openai/v1 is the GA model surface.
    base_url = f"{RESOURCE_ENDPOINT.rstrip('/')}/openai/v1"

    client = OpenAI(base_url=base_url, api_key=API_KEY)

    data_url = image_data_url(IMAGE_PATH)
    print(f"Model deployment: {MODEL_DEPLOYMENT}")
    print(f"Image: {IMAGE_PATH}  ({len(data_url):,} characters encoded)\n")
    pause()

    # --- Full detail ---------------------------------------------------------
    print("=== Triage note, detail=high ===")
    report(triage(client, data_url, detail="high"))
    pause()

    # --- Low detail ----------------------------------------------------------
    print("=== Triage note, detail=low ===")
    report(triage(client, data_url, detail="low"))


if __name__ == "__main__":
    main()
