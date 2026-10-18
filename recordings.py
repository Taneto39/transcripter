"""Recording names shared by every pipeline step.

A recording is identified by its file name with the extension folded in
(``meeting.m4a`` -> ``meeting_m4a``), so two files that differ only by
extension never share chunk, transcript or CSV names.
"""
import glob
import os

from config import DATA_DIR

CHUNK_MARKER = "_chunk_"
OUTPUT_EXTENSIONS = (".csv", ".json", ".log", ".txt")


def resolve_record(pattern):
    """Resolve a path or glob (e.g. 20260816*) to exactly one file, looking in data/ when cwd has no match."""
    if os.path.exists(pattern):
        return pattern
    matches = glob.glob(pattern) or glob.glob(os.path.join(DATA_DIR, pattern))
    # Never pick up result or log files that sit next to recordings.
    matches = [m for m in matches if os.path.isfile(m) and not m.lower().endswith(OUTPUT_EXTENSIONS)]
    if not matches:
        raise SystemExit(f"No file matches '{pattern}'")
    if len(matches) > 1:
        raise SystemExit(
            f"{len(matches)} files match '{pattern}', be more specific:\n" + "\n".join(f"  {m}" for m in matches)
        )
    return matches[0]


def recording_name(path):
    stem, ext = os.path.splitext(os.path.basename(path))
    return f"{stem}_{ext.lstrip('.').lower()}" if ext else stem


def recordings_in(blob_names, prefix):
    """Distinct recording names among ``{prefix}{recording}_chunk_...`` blobs, sorted."""
    names = set()
    for name in blob_names:
        if name.startswith(prefix) and CHUNK_MARKER in name:
            names.add(name[len(prefix):].split(CHUNK_MARKER)[0])
    return sorted(names)


def choose_recording(recordings, query=None, enter_picks_first=False):
    """Pick one recording: exact name, or a unique substring match (e.g. a date), or interactively."""
    if query:
        if query in recordings:
            return query
        recordings = [r for r in recordings if query in r]
        if len(recordings) == 1:
            return recordings[0]
    if not recordings:
        raise SystemExit(f"No recording matches '{query}'" if query else "No recordings found")

    print("Recordings:")
    for i, name in enumerate(recordings, 1):
        print(f"  [{i}] {name}")
    choice = input("Pick a number (Enter = 1): " if enter_picks_first else "Pick a number: ").strip()
    if not choice and enter_picks_first:
        return recordings[0]
    if choice.isdigit() and 1 <= int(choice) <= len(recordings):
        return recordings[int(choice) - 1]
    raise SystemExit(f"Invalid choice '{choice}'")
