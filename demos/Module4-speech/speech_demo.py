"""
Module 4 - speech to text and text to speech with the Speech SDK.

Transcribes a spoken support message from a wav file, then synthesizes a
reply with a selected neural voice.

Before running:
    pip install -r requirements.txt
"""

import os
import sys

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

RESOURCE_ENDPOINT = os.getenv("FOUNDRY_RESOURCE_ENDPOINT")
API_KEY = os.getenv("FOUNDRY_API_KEY")

INPUT_WAV = "support_call.wav"

# Voice names follow locale-name-Neural. Swap this to hear a different voice;
VOICE = "en-US-AvaMultilingualNeural"

PAUSE = os.getenv("DEMO_PAUSE", "1") != "0"

def pause() -> None:
    """Hold one section of output on screen until Enter."""
    if PAUSE:
        input("  -- Enter to continue --")
        print()


def transcribe(speech_config: speechsdk.SpeechConfig) -> str:
    """Speech to text from a wav file."""
    if not os.path.exists(INPUT_WAV):
        sys.exit(f"{INPUT_WAV} not found")

    speech_config.speech_recognition_language = "en-US"

    # Audio comes from a file here. use_default_microphone=True is the
    # live-input swap, and it is the only line that changes.
    audio_config = speechsdk.audio.AudioConfig(filename=INPUT_WAV)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    print("=== Speech to text ===")
    print(f"  source: {INPUT_WAV}")

    # recognize_once_async() also handles one utterance and stops at the first
    # long silence, so it suits a short prompt. Longer audio needs continuous 
    # recognition or batch transcription.
    result = recognizer.recognize_once_async().get()

    #  A failed recognition returns a result object with an empty .text rather than raising an error.
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"  reason: {result.reason}")
        print(f"  text:   {result.text}")
        return result.text

    if result.reason == speechsdk.ResultReason.NoMatch:
        sys.exit(f"  No speech recognized: {result.no_match_details}")

    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        message = f"  Canceled: {details.reason}"
        if details.reason == speechsdk.CancellationReason.Error:
            message += f"\n  Error details: {details.error_details}"
        sys.exit(message)

    sys.exit(f"  Unexpected reason: {result.reason}")


def synthesize(speech_config: speechsdk.SpeechConfig, text: str) -> None:
    """Text to speech through the default speaker."""
    # The voice is set on the config, not passed to the synthesizer.
    speech_config.speech_synthesis_voice_name = VOICE

    # Swap to AudioOutputConfig(filename="reply.wav") to write a file instead
    # of playing through the speaker.
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    print("=== Text to speech ===")
    print(f"  voice: {VOICE}")
    print(f"  text:  {text}")

    result = synthesizer.speak_text_async(text).get()

    # Different enum member from recognition - SynthesizingAudioCompleted,
    # not RecognizedSpeech. Same discipline: check before moving on.
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"  reason: {result.reason}")
        print(f"  audio:  {len(result.audio_data)} bytes")
        return

    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        message = f"  Canceled: {details.reason}"
        if details.reason == speechsdk.CancellationReason.Error:
            message += f"\n  Error details: {details.error_details}"
        sys.exit(message)

    sys.exit(f"  Unexpected reason: {result.reason}")


def main() -> None:
    # Fail early and legibly
    if not RESOURCE_ENDPOINT or not API_KEY:
        sys.exit(
            "Missing configuration. Set FOUNDRY_RESOURCE_ENDPOINT and "
            "FOUNDRY_API_KEY in the root .env file."
        )

    # The config carries the endpoint and key. One config serves both the
    # recognizer and the synthesizer.
    speech_config = speechsdk.SpeechConfig(
        subscription=API_KEY, endpoint=RESOURCE_ENDPOINT.rstrip("/")
    )

    transcript = transcribe(speech_config)
    pause()

    reply = (
        f"Thanks for contacting Globomantics. I have your message: "
        f"{transcript} A replacement is on the way."
    )
    synthesize(speech_config, reply)


if __name__ == "__main__":
    main()