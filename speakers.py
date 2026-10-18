"""Diarization file format and speaker lookup, shared by diarize.py (writes) and batch_download.py (reads).

No pyannote/torch imports here so the batch env can use it.
"""
import bisect
import csv
import os

from config import OUTPUT_DIR


def diarization_path(recording):
    return os.path.join(OUTPUT_DIR, f"{recording}_diarization.csv")


def write_turns(path, turns):
    """turns: iterable of (start, end, speaker) in seconds of the full recording."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "speaker"])
        for start, end, speaker in turns:
            writer.writerow([round(start, 2), round(end, 2), speaker])


def read_turns(path):
    with open(path, encoding="utf-8-sig") as f:
        return [(float(r["start"]), float(r["end"]), r["speaker"]) for r in csv.DictReader(f)]


class SpeakerLookup:
    """Speaker at a point in time; times between turns go to the nearest turn."""

    def __init__(self, turns):
        self.turns = sorted(turns)
        self.starts = [t[0] for t in self.turns]

    def at(self, t):
        if not self.turns:
            return "Unknown"
        i = bisect.bisect_right(self.starts, t) - 1
        candidates = [self.turns[j] for j in (i, i + 1) if 0 <= j < len(self.turns)]
        # Overlapping turns: prefer one that contains t, else the closest edge.
        return min(candidates, key=lambda turn: 0 if turn[0] <= t <= turn[1] else min(abs(t - turn[0]), abs(t - turn[1])))[2]
