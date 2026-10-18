# env=transcripter
"""Speaker diarization with pyannote; writes data/{recording}_diarization.csv (start, end, speaker).

The batch pipeline's download step picks this file up automatically and adds a speaker column.
"""
import argparse
import os
import warnings

import torch
from pyannote.audio import Pipeline
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

import config  # noqa: F401  loads .env (HF_TOKEN)
from recordings import recording_name, resolve_record
from speakers import diarization_path, write_turns

# Harmless pyannote noise: TF32 notice, and std() over single-frame segments.
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=r".*std\(\): degrees of freedom.*")


def load_pipeline():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN not found. Set it in .env (see .env.example) or as an environment variable, e.g.\n"
            "  PowerShell: $env:HF_TOKEN = \"hf_...\""
        )
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    # The GPU does the work; extra CPU threads only spin-wait (same wall time, ~8x the CPU at 14 threads).
    torch.set_num_threads(1)
    return pipeline.to(torch.device("cuda"))


def load_audio(record):
    """Decode a file once to 16 kHz mono; returns torchcodec AudioSamples (.data is a (1, n) tensor)."""
    return AudioDecoder(record, sample_rate=16000, num_channels=1).get_all_samples()


def diarize(pipeline, record, samples=None):
    """Run diarization on one file; returns [(start, end, speaker), ...] sorted by start."""
    print("Diarizing...")
    current = {"step": None, "bar": None}

    # pyannote runs several steps (segmentation, embeddings, ...), each reporting its own progress.
    def hook(step_name, step_artifact, file=None, total=None, completed=None):
        if completed is None or total is None:
            return
        if step_name != current["step"]:
            if current["bar"]:
                current["bar"].close()
            current["step"], current["bar"] = step_name, tqdm(total=total, desc=step_name)
        current["bar"].n = min(completed, total)
        current["bar"].refresh()

    # Decode once: given a path, pyannote re-decodes the file for every embedding window (~40% slower).
    if samples is None:
        samples = load_audio(record)
    try:
        output = pipeline({"waveform": samples.data, "sample_rate": samples.sample_rate}, hook=hook)
    finally:
        if current["bar"]:
            current["bar"].close()

    # exclusive_speaker_diarization (pyannote 4.x) has no overlapping turns, which is what word alignment needs.
    annotation = getattr(output, "exclusive_speaker_diarization", None) or output.speaker_diarization
    return sorted((turn.start, turn.end, speaker) for turn, _, speaker in annotation.itertracks(yield_label=True))


def save(record, turns):
    path = diarization_path(recording_name(record))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_turns(path, turns)
    print(f"Saved diarization: {path} ({len(turns)} turns, {len({t[2] for t in turns})} speakers)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker diarization of one recording (pyannote, local GPU)")
    parser.add_argument("record", help="audio/video path or glob, e.g. 20261018*.m4a (also searched in data/)")
    args = parser.parse_args()

    record = resolve_record(args.record)
    print(f"File: {record}")
    save(record, diarize(load_pipeline(), record))
