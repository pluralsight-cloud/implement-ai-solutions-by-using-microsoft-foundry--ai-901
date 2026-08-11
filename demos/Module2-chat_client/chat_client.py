"""
Module 2 - a minimal chat client using Foundry's OpenAI-compatible API.

Multi-turn state is handled server-side: each turn passes only the new
message plus the previous response ID, and the service reconstructs the
prior context.

Before running:
    pip install -r requirements.txt
    Copy .env.example to .env and add your Foundry resource endpoint,
    resource key, and model deployment name.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

RESOURCE_ENDPOINT = os.getenv("FOUNDRY_RESOURCE_ENDPOINT")
API_KEY = os.getenv("FOUNDRY_API_KEY")
MODEL_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT_NAME")

INSTRUCTIONS = (
    "You are a support assistant for Globomantics, a company that sells "
    "outdoor equipment. Answer in two sentences or fewer. If you do not know "
    "the answer, say so rather than guessing."
)


def main() -> None:
    # Fail early and legibly rather than throwing a KeyError on camera.
    if not RESOURCE_ENDPOINT or not API_KEY or not MODEL_DEPLOYMENT:
        sys.exit(
            "Missing configuration. Copy .env.example to .env and fill in "
            "FOUNDRY_RESOURCE_ENDPOINT, FOUNDRY_API_KEY, and "
            "MODEL_DEPLOYMENT_NAME."
        )

    # The OpenAI-compatible v1 route does not use an api-version query
    # parameter. The OpenAI SDK adds /responses to this base URL.
    # The resource key authenticates the request, so this demo does not
    # depend on `az login` or Foundry data-plane RBAC roles.
    client = OpenAI(
        api_key=API_KEY,
        base_url=f"{RESOURCE_ENDPOINT.rstrip('/')}/openai/v1/",
    )

    print(f"Connected. Model deployment: {MODEL_DEPLOYMENT}")
    print("Ask a question. 'reset' starts a new thread, 'quit' exits.\n")

    # The only state we hold is a handle to the last response. The service
    # stores the transcript (store defaults to true).
    previous_id = None

    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break
        if user_input.lower() == "reset":
            previous_id = None
            print("\n-- context cleared --\n")
            continue

        # `model=` takes the DEPLOYMENT name, not the model name.
        #
        # `instructions` must be resent every turn: previous_response_id
        # carries the message history forward, but NOT the top-level
        # instructions from the earlier response.
        #
        # `input` is only the new message. Passing the whole transcript here
        # as well would duplicate context the service already has.
        response = client.responses.create(
            model=MODEL_DEPLOYMENT,
            instructions=INSTRUCTIONS,
            input=user_input,
            previous_response_id=previous_id,
        )

        # Responses API returns .output_text.
        # Chat Completions would be response.choices[0].message.content
        print(f"\nmodel > {response.output_text}\n")

        # Carry the handle, not the transcript.
        previous_id = response.id


if __name__ == "__main__":
    main()
