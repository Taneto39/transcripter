"""Shared settings for every script.

Machine-specific values (GCP key file, project, bucket, HF token) come from .env (see .env.example),
so nothing personal lives in the code. Inputs and working files (source recordings, chunks, transcript
JSON, operations.log) live in DATA_DIR (data/); results (transcript and diarization CSVs) in OUTPUT_DIR
(output/). Neither is committed.

CHUNK_SECONDS must not change once chunks exist: batch_download.py derives each chunk's offset from it.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def load_dotenv(path=os.path.join(BASE_DIR, ".env")):
    """Load KEY=VALUE lines from .env without overriding variables already set."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()

LOCATION = "us"  # chirp_3 only runs in us/eu
API_ENDPOINT = f"{LOCATION}-speech.googleapis.com"

GCS_AUDIO_PREFIX = "audio-files/"
GCS_TRANSCRIPTS_PREFIX = "transcripts/"

CHUNK_SECONDS = 1200  # 20 minutes
PAUSE_THRESHOLD = 1  # seconds of silence that start a new CSV segment
MAX_FILES_PER_REQUEST = 5  # Speech-to-Text v2 batch_recognize limit

CHUNK_DIR = os.path.join(DATA_DIR, "chunks")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "transcripts")
LOG_FILE = os.path.join(DATA_DIR, "operations.log")

# PROJECT_ID (string) is used in requests; PROJECT_NUMBER (numeric) is what Google puts in operation
# names, needed to rebuild a full operation name from a short id. They are not interchangeable.
_GCP_SETTINGS = ("KEY_FILE", "PROJECT_ID", "PROJECT_NUMBER", "BUCKET")


def __getattr__(name):
    # Resolved lazily so scripts that never touch GCP (diarize.py, transcripter.py) don't need these set.
    if name in _GCP_SETTINGS:
        value = os.environ.get(name)
        if not value:
            raise SystemExit(f"{name} is not set; add it to .env (see .env.example)")
        return value
    if name == "OUTPUT_URI":
        return f"gs://{__getattr__('BUCKET')}/{GCS_TRANSCRIPTS_PREFIX}"
    raise AttributeError(f"module 'config' has no attribute '{name}'")
