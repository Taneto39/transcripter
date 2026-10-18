import argparse
import csv
import json
import os

from google.cloud import storage
from google.oauth2 import service_account

from config import (
    BUCKET,
    CHUNK_SECONDS,
    DOWNLOAD_DIR,
    GCS_TRANSCRIPTS_PREFIX,
    KEY_FILE,
    OUTPUT_DIR,
    PAUSE_THRESHOLD,
    PROJECT_ID,
)
from recordings import CHUNK_MARKER, choose_recording, recordings_in
from speakers import SpeakerLookup, diarization_path, read_turns


def download(query=None):
    """Download the transcript JSON of one recording into DOWNLOAD_DIR; returns (recording, paths).

    query: recording name, or any unique part of it (e.g. a date); prompts when ambiguous or missing.
    """
    credentials = service_account.Credentials.from_service_account_file(KEY_FILE)
    storage_client = storage.Client(credentials=credentials, project=PROJECT_ID)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    blobs = [b for b in storage_client.list_blobs(BUCKET, prefix=GCS_TRANSCRIPTS_PREFIX) if b.name.endswith(".json")]
    recording = choose_recording(recordings_in([b.name for b in blobs], GCS_TRANSCRIPTS_PREFIX), query)

    chunk_prefix = f"{GCS_TRANSCRIPTS_PREFIX}{recording}{CHUNK_MARKER}"
    # A chunk sent more than once (e.g. resent after a failure) has one JSON per run; keep the newest.
    latest = {}
    for b in blobs:
        if b.name.startswith(chunk_prefix):
            chunk = b.name[len(chunk_prefix):].split("_")[0]
            if chunk not in latest or b.updated > latest[chunk].updated:
                latest[chunk] = b
    targets = sorted(latest.values(), key=lambda b: b.name)
    skipped = sum(b.name.startswith(chunk_prefix) for b in blobs) - len(targets)

    print(f"{recording}: {len(targets)} files, downloading..." + (f" ({skipped} older duplicates skipped)" if skipped else ""))
    paths = []
    for blob in targets:
        filename = os.path.basename(blob.name)
        dest = os.path.join(DOWNLOAD_DIR, filename)
        blob.download_to_filename(dest)
        print(f"  {filename}")
        paths.append(dest)
    return recording, paths


MAX_WORD_SECONDS = 3


def parse_offset(s):
    return float(s.replace("s", ""))


def process(json_paths, recording):
    """Convert one recording's transcript JSON into {recording}_transcript.csv.

    Columns are start, text; plus speaker when data/{recording}_diarization.csv exists (from diarize.py),
    in which case a speaker change also starts a new segment.
    """
    turns_path = diarization_path(recording)
    speakers = SpeakerLookup(read_turns(turns_path)) if os.path.exists(turns_path) else None
    print(f"Speakers from {turns_path}" if speakers else f"No {os.path.basename(turns_path)}; writing without speakers")

    rows = []
    for path in sorted(json_paths):
        # Each chunk's timestamps restart at 0; the chunk index in the file name gives its offset.
        basename = os.path.basename(path)
        chunk_index = int(basename.split("_chunk_")[1].split("_")[0])
        chunk_offset = chunk_index * CHUNK_SECONDS

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for result in data.get("results", []):
            words = result["alternatives"][0].get("words", [])
            if not words:
                continue

            segment_words = []
            segment_start = None
            segment_speaker = None
            prev_end = None

            def flush():
                row = {"start": round(segment_start, 2), "text": "".join(segment_words)}
                if speakers:
                    row["speaker"] = segment_speaker
                rows.append(row)

            for w in words:
                start = chunk_offset + parse_offset(w.get("startOffset", "0s"))
                # chirp_3 occasionally returns a word lasting minutes or hours; cap it so it can't swallow pauses.
                end = min(chunk_offset + parse_offset(w.get("endOffset", "0s")), start + MAX_WORD_SECONDS)
                speaker = speakers.at(start) if speakers else None

                if segment_words and ((start - prev_end) > PAUSE_THRESHOLD or speaker != segment_speaker):
                    flush()
                    segment_words = []

                if not segment_words:
                    segment_start = start
                    segment_speaker = speaker

                segment_words.append(w["word"])
                prev_end = end

            if segment_words:
                flush()

    rows.sort(key=lambda r: r["start"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{recording}_transcript.csv")
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "speaker", "text"] if speakers else ["start", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download one recording's transcript JSON from GCS and convert it to CSV")
    parser.add_argument("recording", nargs="?", help="recording name or part of it (e.g. 20261018); omit to pick from a list")
    args = parser.parse_args()

    recording, json_paths = download(args.recording)
    process(json_paths, recording)
