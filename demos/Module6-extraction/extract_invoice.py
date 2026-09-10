"""
Module 6 - extracting fields from a document with a prebuilt analyzer.

Runs the prebuilt invoice analyzer over a Globomantics invoice and reads back
the markdown the analyzer produced, the extracted fields with their confidence
scores, and what the run cost.

Before running:
    pip install -r requirements.txt
"""

import os
import sys

from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.ai.contentunderstanding.models import AnalysisInput
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

RESOURCE_ENDPOINT = os.getenv("FOUNDRY_RESOURCE_ENDPOINT")
API_KEY = os.getenv("FOUNDRY_API_KEY")
MODEL_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT_NAME")

EMBEDDING_DEPLOYMENT = "text-embedding-3-large"

ANALYZER_ID = "prebuilt-invoice"
DOCUMENT_PATH = "globomantics_invoice_generated_by_claude.pdf"

MARKDOWN_LINES = 16

# Set DEMO_PAUSE=0 in the environment to run straight through.
PAUSE = os.getenv("DEMO_PAUSE", "1") != "0"


def pause() -> None:
    """Hold one section of output on screen until Enter."""
    if PAUSE:
        input("  -- Enter to continue --")
        print()


def start_analysis(client: ContentUnderstandingClient, path: str):
    """Send the local file and hand back the poller for the running job."""
    if not os.path.exists(path):
        sys.exit(f"No document at {path}.")

    with open(path, "rb") as document:
        file_bytes = document.read()

    # Analysis is a long-running operation, so this starts the job and returns
    # a poller rather than a result.
    return client.begin_analyze(
        ANALYZER_ID,
        inputs=[
            AnalysisInput(
                data=file_bytes,
                mime_type="application/pdf",
                name=os.path.basename(path),
            )
        ],
        # Prebuilt analyzers ask for models by alias, not by model name, so
        # these two keys are fixed and the values are deployments of mine.
        # Leave this off and the analyzer uses the resource defaults instead.
        model_deployments={
            "prebuilt-analyzer-completion": MODEL_DEPLOYMENT,
            "prebuilt-analyzer-embedding": EMBEDDING_DEPLOYMENT,
        },
    )


def show_field(name: str, field, indent: str = "") -> None:
    """Print one extracted field next to the score the service put on it."""
    confidence = "  --" if field.confidence is None else f"{field.confidence:.2f}"
    print(f"{indent}{confidence}  {name:<22} {field.value}")


def show_array(field) -> None:
    """Walk an array field, which holds one entry per row on the invoice."""
    for number, row in enumerate(field.value, start=1):
        print(f"  Row {number}")

        # A row is usually an object of its own fields, but an analyzer can
        # return an array of plain values instead.
        if row.type == "object":
            for name, row_field in row.value.items():
                show_field(name, row_field, indent="    ")
        else:
            show_field(row.type, row, indent="    ")


def show_usage(usage) -> None:
    """Print what the run consumed."""
    if usage is None:
        print("  No usage reported.")
        return

    for label, value in (
        ("Pages, minimal", usage.document_pages_minimal),
        ("Pages, basic", usage.document_pages_basic),
        ("Pages, standard", usage.document_pages_standard),
        ("Contextualization tokens", usage.contextualization_tokens),
    ):
        if value is not None:
            print(f"  {label:<26} {value}")

    for model, count in (usage.tokens or {}).items():
        print(f"  {model:<26} {count}")


def main() -> None:
    if not RESOURCE_ENDPOINT or not API_KEY or not MODEL_DEPLOYMENT:
        sys.exit(
            "Missing configuration. Copy .env.example to .env and fill in "
            "FOUNDRY_RESOURCE_ENDPOINT, FOUNDRY_API_KEY, and "
            "MODEL_DEPLOYMENT_NAME."
        )

    # Content Understanding is a Foundry Tool, so it takes the resource
    # endpoint as it comes - no /openai/v1 suffix like the model calls.
    client = ContentUnderstandingClient(
        endpoint=RESOURCE_ENDPOINT,
        credential=AzureKeyCredential(API_KEY),
    )

    print(f"Analyzer: {ANALYZER_ID}")
    print(f"Document: {DOCUMENT_PATH}")
    print(f"Models:   {MODEL_DEPLOYMENT}, {EMBEDDING_DEPLOYMENT}\n")
    pause()

    poller = start_analysis(client, DOCUMENT_PATH)
    print("Analyzing", end="", flush=True)
    result = poller.result()
    print(" ... done.\n")

    # One entry per input file, so a single document lands at index zero.
    content = result.contents[0]

    # --- What the analyzer read ----------------------------------------------
    print("=== Markdown content ===")
    lines = content.markdown.splitlines()
    print("\n".join(lines[:MARKDOWN_LINES]))
    print(f"  [{len(lines) - MARKDOWN_LINES} more lines]\n")
    pause()

    # --- What the analyzer pulled out ----------------------------------------
    print("=== Extracted fields ===")
    arrays = {}
    for name, field in content.fields.items():
        # Table-shaped fields come back as arrays, so hold them for their own
        # section rather than printing a nested object on one line.
        if field.type == "array":
            arrays[name] = field
        else:
            show_field(name, field)
    print()
    pause()

    for name, field in arrays.items():
        print(f"=== {name} ===")
        show_array(field)
        print()
        pause()

    # --- What it cost --------------------------------------------------------
    # Usage hangs off the poller
    print("=== Usage ===")
    show_usage(poller.usage)


if __name__ == "__main__":
    main()