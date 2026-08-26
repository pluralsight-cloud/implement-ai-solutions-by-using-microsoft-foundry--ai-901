"""
Module 4 - text analysis with a dedicated Foundry Tools client.

Runs three analyses over one Globomantics support message: sentiment
(document and per sentence), named entity recognition, and PII detection.

Note the endpoint. This is the same Foundry resource and the same key the
chat client used, but the Language surface sits at the bare resource
endpoint - no /openai/v1, and no /api/projects/... either.

Before running:
    pip install -r requirements.txt
"""

import os
import sys

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

RESOURCE_ENDPOINT = os.getenv("FOUNDRY_RESOURCE_ENDPOINT")
API_KEY = os.getenv("FOUNDRY_API_KEY")

SUPPORT_MESSAGE = (
    "Hi, this is Dana Reyes. I ordered the Summit 45L backpack from "
    "Globomantics on March 3rd, and shipping was impressively fast. "
    "Unfortunately the left shoulder strap tore on my second hike, which is "
    "disappointing for a pack at this price. Reach me at (206) 555-0142 or "
    "dana.reyes@example.com to sort out a replacement - my order number "
    "is GLM-88231."
)


def main() -> None:
    # Fail early and legibly rather than throwing on camera.
    if not RESOURCE_ENDPOINT or not API_KEY:
        sys.exit(
            "Missing configuration. Set FOUNDRY_RESOURCE_ENDPOINT and "
            "FOUNDRY_API_KEY in the root .env file."
        )

    # AzureKeyCredential wraps the key. Swapping in DefaultAzureCredential
    # here is the only change the Entra ID path needs - this client accepts
    # either one.
    client = TextAnalyticsClient(
        endpoint=RESOURCE_ENDPOINT.rstrip("/"),
        credential=AzureKeyCredential(API_KEY),
    )

    print(f"\nMessage:\n  {SUPPORT_MESSAGE}\n")

    # --- Sentiment -------------------------------------------------------
    # documents= takes a LIST, and results come back as a list in the same
    # order - one result per document. The service reads each character as its own
    # document. One document in, so index 0 back out.
    sentiment = client.analyze_sentiment(documents=[SUPPORT_MESSAGE])[0]

    # A document that fails does NOT raise. It arrives in the list as a
    # DocumentError, so check is_error before reading anything off it.
    if sentiment.is_error:
        sys.exit(f"Service error {sentiment.error.code}: {sentiment.error.message}")

    # The document label is one of positive, negative, neutral, or mixed.
    scores = sentiment.confidence_scores
    print("=== Sentiment ===")
    print(
        f"document: {sentiment.sentiment}  "
        f"(pos {scores.positive:.2f} / neu {scores.neutral:.2f} / "
        f"neg {scores.negative:.2f})"
    )

    # Per-sentence results are why a dedicated client can be more useful than asking a model
    # for a one-word answer: the breakdown comes back structured.
    for sentence in sentiment.sentences:
        print(f"  [{sentence.sentiment:>8}] {sentence.text}")

    # --- Named entities --------------------------------------------------
    entities = client.recognize_entities(documents=[SUPPORT_MESSAGE])[0]

    print("\n=== Entities ===")
    for entity in entities.entities:
        # Subcategory is often empty - Organization has none, DateTime does.
        label = entity.category
        if entity.subcategory:
            label += f"/{entity.subcategory}"
        print(f"  {entity.text:<24} {label:<22} {entity.confidence_score:.2f}")

    # --- PII -------------------------------------------------------------
    # A separate call and a separate model from recognize_entities. This one
    # looks for personal information specifically, and hands back a redacted
    # copy of the text alongside the findings.
    pii = client.recognize_pii_entities(documents=[SUPPORT_MESSAGE])[0]

    print("\n=== PII ===")
    for entity in pii.entities:
        print(
            f"  {entity.text:<24} {entity.category:<16} "
            f"{entity.confidence_score:.2f}"
        )

    print(f"\nRedacted:\n  {pii.redacted_text}\n")


if __name__ == "__main__":
    main()
