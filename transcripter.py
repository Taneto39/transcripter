# env=transcripter
import argparse
import csv
import os

from faster_whisper import WhisperModel
from tqdm import tqdm

import diarize
from config import BASE_DIR, OUTPUT_DIR
from recordings import recording_name, resolve_record
from speakers import SpeakerLookup

parser = argparse.ArgumentParser(description="Transcribe and diarize one file locally (faster-whisper + pyannote)")
parser.add_argument("record", help="audio/video path or glob, e.g. 20260816* (also searched in data/)")
args = parser.parse_args()

record = resolve_record(args.record)
print(f"File: {record}")

pipeline = diarize.load_pipeline()

WHISPER_MODEL = os.path.join(BASE_DIR, "models", "typhoon-whisper-large-v3-ct2")
whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")

turns = diarize.diarize(pipeline, record)
diarize.save(record, turns)  # reused by the batch pipeline's download step
speakers = SpeakerLookup(turns)

# Transcribe
print("Transcribing...")
segments, info = whisper.transcribe(record, language="th")
total_duration = round(info.duration)

# Same naming as the batch pipeline ({recording}_transcript.csv) plus _local so the two never overwrite each other.
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, f"{recording_name(record)}_transcript_local.csv")

with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["start", "end", "speaker", "text"])

    with tqdm(total=total_duration, unit="sec", desc="Progress") as pbar:
        for segment in segments:
            writer.writerow([
                round(segment.start, 2),
                round(segment.end, 2),
                speakers.at(segment.start),
                segment.text.strip()
            ])
            # Track the absolute position (segment.end), not summed (end - start): speech spans skip
            # silence, so summing them under-reports progress badly and makes the ETA look like a slow GPU.
            pbar.n = min(round(segment.end), total_duration)
            pbar.refresh()

print(f"Saved: {output_path}")
