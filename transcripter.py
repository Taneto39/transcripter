# env=transcripter
"""Local pipeline: pyannote diarization, then Typhoon Whisper on each speaker turn.

Typhoon emits no usable timestamps (one segment per 30 s window) and loops on long windows, so the
audio is cut at diarization turns instead: each piece is one speaker, at most MAX_PIECE_SECONDS long,
and its time and speaker come from the diarization rather than from whisper.
"""
import argparse
import bisect
import csv
import os
import zlib

from faster_whisper import BatchedInferencePipeline, WhisperModel
from tqdm import tqdm

import diarize
from config import BASE_DIR, OUTPUT_DIR
from recordings import recording_name, resolve_record
from speakers import diarization_path, read_turns

WHISPER_MODEL = os.path.join(BASE_DIR, "models", "typhoon-whisper-large-v3-ct2")
MERGE_GAP_SECONDS = 1.0    # join turns of the same speaker separated by less than this
MAX_PIECE_SECONDS = 28.0   # whisper reads at most 30 s per window
MIN_PIECE_SECONDS = 0.5    # shorter pieces mostly come back as made-up words
PAD_SECONDS = 0.25         # widen each piece so a turn's first syllable isn't clipped
MAX_COMPRESSION = 3.0      # zlib ratio of the UTF-8 text; loops score far above (17+), long Thai speech below ~2.8
BATCH_SIZE = 8
# Output sometimes starts mid-character: a lone Thai vowel/tone mark or a replacement char.
LEADING_JUNK = "�ะาำัิีึืฺุู็่้๊๋์ํ๎. "


def clean(text):
    """Strip a broken leading character; return '' for looping (highly compressible) text."""
    text = text.strip().lstrip(LEADING_JUNK)
    data = text.encode()
    if data and len(data) / len(zlib.compress(data)) > MAX_COMPRESSION:
        return ""
    return text


def make_pieces(turns):
    """Merge and split diarization turns into [start, end, speaker] pieces whisper can take whole."""
    pieces = []
    for start, end, speaker in turns:
        if (pieces and pieces[-1][2] == speaker and start - pieces[-1][1] < MERGE_GAP_SECONDS
                and end - pieces[-1][0] <= MAX_PIECE_SECONDS):
            pieces[-1][1] = end
            continue
        while end - start > MAX_PIECE_SECONDS:
            pieces.append([start, start + MAX_PIECE_SECONDS, speaker])
            start += MAX_PIECE_SECONDS
        pieces.append([start, end, speaker])
    return [p for p in pieces if p[1] - p[0] >= MIN_PIECE_SECONDS]


def main():
    parser = argparse.ArgumentParser(description="Transcribe and diarize one file locally (Typhoon Whisper + pyannote)")
    parser.add_argument("record", help="audio/video path or glob, e.g. 20260816* (also searched in data/)")
    args = parser.parse_args()

    record = resolve_record(args.record)
    print(f"File: {record}")

    # Decode once with torchcodec and share it: faster-whisper's own PyAV decoding breaks on av >= 17.
    samples = diarize.load_audio(record)

    diarization_file = diarization_path(recording_name(record))
    if os.path.exists(diarization_file):
        print(f"Using existing diarization: {diarization_file} (delete it to diarize again)")
        turns = read_turns(diarization_file)
    else:
        turns = diarize.diarize(diarize.load_pipeline(), record, samples)
        diarize.save(record, turns)  # reused by the batch pipeline's download step

    audio = samples.data[0].numpy()
    duration = len(audio) / samples.sample_rate
    pieces = make_pieces(turns)
    clips = [{"start": max(start - PAD_SECONDS, 0), "end": min(end + PAD_SECONDS, duration)} for start, end, _ in pieces]
    starts = [clip["start"] for clip in clips]
    print(f"Transcribing {len(pieces)} speaker turns...")

    whisper = BatchedInferencePipeline(WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16"))
    segments, _ = whisper.transcribe(
        audio,
        language="th",
        clip_timestamps=clips,
        without_timestamps=True,
        beam_size=5,
        batch_size=BATCH_SIZE,
    )

    # Same naming as the batch pipeline ({recording}_transcript.csv) plus _local so the two never overwrite each other.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{recording_name(record)}_transcript_local.csv")

    texts = {}
    for segment in tqdm(segments, total=len(pieces), unit="turn", desc="Progress"):
        # Each segment starts at its clip's offset; map it back to the piece it came from.
        i = max(bisect.bisect_right(starts, segment.start + 0.01) - 1, 0)
        texts.setdefault(i, []).append(clean(segment.text))

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "speaker", "text"])
        for i, (start, end, speaker) in enumerate(pieces):
            text = " ".join(t for t in texts.get(i, []) if t)
            if text:
                writer.writerow([round(start, 2), round(end, 2), speaker, text])

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
