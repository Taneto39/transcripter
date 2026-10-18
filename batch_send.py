import argparse
import os
from datetime import datetime

from google.cloud import speech_v2, storage
from google.cloud.speech_v2.types import cloud_speech
from google.oauth2 import service_account

from config import (
    API_ENDPOINT,
    BUCKET,
    GCS_AUDIO_PREFIX,
    KEY_FILE,
    LOCATION,
    LOG_FILE,
    MAX_FILES_PER_REQUEST,
    OUTPUT_URI,
    PROJECT_ID,
)
from recordings import CHUNK_MARKER, choose_recording, recordings_in


def send_batch(query=None):
    """Send every chunk of one recording as batch_recognize request(s); returns the operation names.

    query: recording name, or any unique part of it (e.g. a date); prompts when ambiguous or missing.
    """
    credentials = service_account.Credentials.from_service_account_file(KEY_FILE)

    storage_client = storage.Client(credentials=credentials, project=PROJECT_ID)
    names = [b.name for b in storage_client.list_blobs(BUCKET, prefix=GCS_AUDIO_PREFIX) if b.name.endswith(".wav")]
    recording = choose_recording(recordings_in(names, GCS_AUDIO_PREFIX), query)

    chunk_prefix = f"{GCS_AUDIO_PREFIX}{recording}{CHUNK_MARKER}"
    chunk_uris = sorted(f"gs://{BUCKET}/{n}" for n in names if n.startswith(chunk_prefix))
    print(f"{recording}: {len(chunk_uris)} chunks")

    client = speech_v2.SpeechClient(
        credentials=credentials,
        client_options={"api_endpoint": API_ENDPOINT},
    )

    # noinspection PyTypeChecker
    recognition_config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["th-TH"],
        model="chirp_3",
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
        ),
    )

    operations = []
    for i in range(0, len(chunk_uris), MAX_FILES_PER_REQUEST):
        batch_uris = chunk_uris[i:i + MAX_FILES_PER_REQUEST]

        # noinspection PyTypeChecker
        request = cloud_speech.BatchRecognizeRequest(
            recognizer=f"projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/_",
            config=recognition_config,
            files=[cloud_speech.BatchRecognizeFileMetadata(uri=uri) for uri in batch_uris],
            recognition_output_config=cloud_speech.RecognitionOutputConfig(
                gcs_output_config=cloud_speech.GcsOutputConfig(uri=OUTPUT_URI),
            ),
            processing_strategy=cloud_speech.BatchRecognizeRequest.ProcessingStrategy.DYNAMIC_BATCHING,
        )

        operation = client.batch_recognize(request=request)
        operations.append(operation.operation.name)
        print(f"Sent ({len(batch_uris)} chunks): {operation.operation.name}")

        # Log each job right away so the operation name survives a closed terminal.
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            timestamp = datetime.now().isoformat(timespec="seconds")
            f.write(f"{timestamp}\t{recording}\t{len(batch_uris)} chunks\t{operation.operation.name}\n")

    print(f"\n{len(operations)} operation(s), logged to {LOG_FILE}")
    print(f"Jobs run in the cloud; it's fine to shut down. Check with: python pipeline.py status {recording}")
    return operations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a batch transcribe job for one recording to Google Cloud Speech-to-Text")
    parser.add_argument("recording", nargs="?", help="recording name or part of it (e.g. 20261018); omit to pick from a list")
    args = parser.parse_args()
    send_batch(args.recording)
