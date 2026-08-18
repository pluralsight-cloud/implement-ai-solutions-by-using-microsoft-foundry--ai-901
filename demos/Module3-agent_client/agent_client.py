"""Talk to a Foundry prompt agent from a client application.

Connects by name to the agent created in the portal, opens a conversation,
and sends turns against it from a prompt loop.

Requires Microsoft Entra ID authentication: run `az login` first.
"""

import os
import sys

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
AGENT_NAME = "Globomantics-Support"


def main():
    if not PROJECT_ENDPOINT or not AGENT_NAME:
        sys.exit(
            "Missing configuration. Add FOUNDRY_PROJECT_ENDPOINT and "
            "FOUNDRY_AGENT_NAME to .env."
        )

    # DefaultAzureCredential picks up the `az login` session. The project
    # client only accepts a token credential, so there is no key path.
    credential = DefaultAzureCredential()

    # AIProjectClient talks to the PROJECT endpoint - the long form that
    # ends in /api/projects/<project-name>.
    project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential)

    # Passing agent_name scopes the client to one agent. Calls made with it
    # run against that agent's instructions, model, and tools.
    client = project.get_openai_client(agent_name=AGENT_NAME)

    # A conversation is a durable, server-side container for the exchange.
    conversation = client.conversations.create()

    print(f"Connected to agent: {AGENT_NAME}")
    print(f"Conversation: {conversation.id}")
    print("Ask a question. 'reset' starts a new conversation, 'quit' exits.\n")

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
            conversation = client.conversations.create()
            print(f"\n-- new conversation: {conversation.id} --\n")
            continue

        # Compare this to the chat client. No model, no instructions - the
        # agent owns both. Only the conversation id and the new message.
        response = client.responses.create(
            conversation=conversation.id,
            input=user_input,
        )

        # Same response shape as a direct model call: .output_text.
        print(f"\nagent > {response.output_text}\n")


if __name__ == "__main__":
    main()