# transcripter

Transcribe (Thai) meeting recordings to CSV (`start`, `speaker`, `text`) with Google Cloud Speech-to-Text v2 (`chirp_3`), with speakers from local pyannote diarization, plus an offline fallback using faster-whisper.

A set of standalone scripts run by hand, in order. No build system or test suite.

## Setup

One environment for every script: Google Cloud clients plus CUDA 12.8 torch, pyannote and faster-whisper (about 3GB, mostly torch).

```bash
conda create -n transcripter python=3.12
conda activate transcripter
conda install -c conda-forge "ffmpeg>=4,<9"   # shared FFmpeg 4-8 for torchcodec; a static or v9 build fails
pip install -r requirements.txt
```

Also required:
- A GCP service account key file, with `KEY_FILE`, `PROJECT_ID`, `PROJECT_NUMBER` and `BUCKET` set in `.env` (copy `.env.example`)
- A GCS bucket you can read and write

## Data layout

Put recordings in `data/`. Working files go there too (`chunks/`, `transcripts/`, `operations.log`); results (transcript and diarization CSVs) go to `output/`. Both folders are gitignored.

## Batch pipeline (Google Cloud)

```bash
conda activate transcripter
python pipeline.py upload [file]        # split into 20-minute PCM WAV chunks (ffmpeg) and upload to GCS
python pipeline.py send [recording]     # submit BatchRecognize jobs (max 5 files per request)
python pipeline.py status [recording]   # check every job of a recording (from operations.log); also takes an operation id
python pipeline.py download [recording] # fetch transcript JSON and write output/{recording}_transcript.csv
```

A *recording* is the file name with its extension folded in (`meeting.m4a` → `meeting_m4a`), so two files that differ only by extension can be transcribed and compared side by side. `send`, `status` and `download` take the full name or any unique part of it (such as a date), and list the candidates when it matches more than one. `upload` skips chunks that are already in the bucket, so an interrupted upload can simply be rerun.

Each step can also be run as its own script (`batch_upload.py`, `batch_send.py`, `operation_progress.py`, `batch_download.py`). Arguments are optional; missing ones are prompted for.

`chirp_3` is only available in the `us` and `eu` locations, so the regional endpoint (`us-speech.googleapis.com`) is used. A `429 RESOURCE_EXHAUSTED` from `status` is a per-minute quota; wait a minute and retry.

### Speaker labels (optional)

`chirp_3` diarization doesn't cover Thai, so speakers come from pyannote on the local GPU:

```bash
conda activate transcripter
python diarize.py <file or glob>        # writes output/{recording}_diarization.csv
```

If that file exists when you run `download`, each word gets the speaker active at its start time, segments also break on speaker changes, and the CSV becomes `start, speaker, text`. Without it you get `start, text`. Diarize each recording separately: two devices recording the same meeting don't share a timeline.

Diarization needs a HuggingFace token with access to gated repos, after accepting the terms of [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0). Put it in `.env` (see `.env.example`) or set `HF_TOKEN`.

## Local pipeline (offline)

```bash
conda activate transcripter
python transcripter.py <file or glob>
```

Writes `output/{recording}_transcript_local.csv` with `start`, `end`, `speaker`, `text`. Uses [typhoon-ai/typhoon-whisper-large-v3](https://huggingface.co/typhoon-ai/typhoon-whisper-large-v3) converted to CTranslate2:

```bash
ct2-transformers-converter --model typhoon-ai/typhoon-whisper-large-v3 --output_dir models/typhoon-whisper-large-v3-ct2 --quantization float16
```

The converter doesn't copy `preprocessor_config.json`. Add one with `"feature_size": 128` (otherwise copied from `openai/whisper-large-v3`) or transcription fails with a feature-shape error. Speakers use the same HuggingFace token as `diarize.py`, and the run also writes `output/{recording}_diarization.csv`.
