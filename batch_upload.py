import argparse
import subprocess
import os
import glob
import tkinter as tk
from tkinter import filedialog

from google.cloud import storage
from google.oauth2 import service_account

from config import BUCKET, CHUNK_DIR, CHUNK_SECONDS, DATA_DIR, GCS_AUDIO_PREFIX, KEY_FILE, PROJECT_ID
from recordings import recording_name


def pick_file():
    """File dialog for an audio/video file; returns the path, or '' if cancelled."""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Choose a recording",
        initialdir=DATA_DIR,
        filetypes=[("Audio/Video files", "*.wav *.mp3 *.m4a *.aac *.mp4 *.mkv"), ("All files", "*.*")],
    )


def chunk_audio(record_path, chunk_dir=CHUNK_DIR, chunk_seconds=CHUNK_SECONDS):
    """Split a recording into PCM WAV chunks with ffmpeg; returns the chunk paths.

    Always transcodes to PCM: copying a non-PCM stream (e.g. AAC from a Teams mp4) into .wav gives unreadable files.
    """
    base_name = recording_name(record_path)
    os.makedirs(chunk_dir, exist_ok=True)

    print("Splitting...")
    subprocess.run([
        "ffmpeg", "-y", "-i", record_path,
        "-vn",
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c:a", "pcm_s16le",
        os.path.join(chunk_dir, f"{base_name}_chunk_%03d.wav"),
    ], check=True)

    return sorted(glob.glob(os.path.join(chunk_dir, f"{base_name}_chunk_*.wav")))


def upload_chunks(chunks):
    """Upload chunks to gs://BUCKET/GCS_AUDIO_PREFIX with the service account; skips ones already there."""
    credentials = service_account.Credentials.from_service_account_file(KEY_FILE)
    bucket = storage.Client(credentials=credentials, project=PROJECT_ID).bucket(BUCKET)

    print(f"{len(chunks)} chunks, uploading to gs://{BUCKET}/{GCS_AUDIO_PREFIX}")
    for chunk in chunks:
        name = os.path.basename(chunk)
        size = os.path.getsize(chunk)
        blob = bucket.blob(GCS_AUDIO_PREFIX + name)
        if blob.exists():
            blob.reload()
            if blob.size == size:
                print(f"  {name} already uploaded, skipping")
                continue
        print(f"  {name} ({size / 1e6:.0f} MB)...", flush=True)
        blob.upload_from_filename(chunk, timeout=3600)


def run(record_path=None):
    """Pick a file (unless given), split it into chunks and upload them."""
    if not record_path:
        record_path = pick_file()
    if not record_path:
        print("No file chosen")
        return

    print(f"File: {record_path}")
    chunks = chunk_audio(record_path)
    upload_chunks(chunks)
    print(f"Uploaded. Recording: {recording_name(record_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a recording into chunks and upload them to GCS")
    parser.add_argument("file", nargs="?", help="audio/video path; omit to open a file dialog")
    args = parser.parse_args()
    run(args.file)
