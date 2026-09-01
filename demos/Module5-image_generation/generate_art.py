"""
Module 5 - generating an image from code.

Generates a hero image for a Globomantics winter camping page that has no
photography yet, then decodes the returned payload and saves it to a file.

Before running:
    pip install -r requirements.txt
"""

import base64
import os
import sys

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()

RESOURCE_ENDPOINT = os.getenv("FOUNDRY_RESOURCE_ENDPOINT")
API_KEY = os.getenv("FOUNDRY_API_KEY")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")

OUTPUT_PATH = "winter_hero.png"

PROMPT = (
    "A wide banner photograph for an outdoor equipment company. A two-person "
    "tent glows from a lantern inside, pitched on packed snow at dusk, pine "
    "forest behind it, mountains on the horizon. Cold blue light, warm light "
    "from the tent, no people, no text."
)

# Set DEMO_PAUSE=0 in the environment to run straight through.
PAUSE = os.getenv("DEMO_PAUSE", "1") != "0"


def pause() -> None:
    """Hold one section of output on screen until Enter."""
    if PAUSE:
        input("  -- Enter to continue --")
        print()


def save(image, path: str) -> None:
    """Write the generated image out, whichever shape came back."""
    # The gpt-image models return base64 in b64_json.
    if image.b64_json:
        image_bytes = base64.b64decode(image.b64_json)
        with open(path, "wb") as image_file:
            image_file.write(image_bytes)
        print(f"Saved {path} ({len(image_bytes):,} bytes)")
    else:
        print(f"Returned a URL rather than bytes: {image.url}")


def main() -> None:
    if not RESOURCE_ENDPOINT or not API_KEY or not MODEL_DEPLOYMENT_NAME:
        sys.exit(
            "Missing configuration. Copy .env.example to .env and fill in "
            "FOUNDRY_RESOURCE_ENDPOINT, FOUNDRY_API_KEY, and "
            "MODEL_DEPLOYMENT_NAME."
        )

    # Same endpoint and same client as the other demos. Only the method and
    # the deployment change.
    base_url = f"{RESOURCE_ENDPOINT.rstrip('/')}/openai/v1"

    client = OpenAI(base_url=base_url, api_key=API_KEY)

    print("=== Request ===")
    print(f"Deployment: {MODEL_DEPLOYMENT_NAME}")
    print(f"Prompt: {PROMPT}\n")
    pause()

    print("=== Generating ===")
    try:
        # Setting size, quality, and landscape here because it is a page banner
        result = client.images.generate(
            model=MODEL_DEPLOYMENT_NAME,
            prompt=PROMPT,
            size="1536x1024",
            quality="medium",
            n=1,
        )
    except BadRequestError as error:
        # A prompt the content filter rejects fails the request outright, so catch it and print the reason.
        sys.exit(f"Request rejected: {error.message}")

    save(result.data[0], OUTPUT_PATH)

    # Image models bill in tokens too, and report them per request.
    if result.usage:
        print(
            f"  input tokens: {result.usage.input_tokens}"
            f"   output tokens: {result.usage.output_tokens}"
        )


if __name__ == "__main__":
    main()
