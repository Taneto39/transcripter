"""One CLI for the four batch pipeline steps, run in order:

    python pipeline.py upload [file]
    python pipeline.py send [recording]
    python pipeline.py status [recording|operation]
    python pipeline.py download [recording]

Each step's script (batch_upload.py, batch_send.py, operation_progress.py, batch_download.py) also runs on its own.
"""
import argparse

import batch_download
import batch_send
import batch_upload
import operation_progress


def main():
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Meeting recording -> CSV transcript via Google Cloud Speech-to-Text (chirp_3)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_upload = sub.add_parser("upload", help="split a recording into chunks and upload them to GCS")
    p_upload.add_argument("file", nargs="?", help="audio/video path; omit to open a file dialog")

    p_send = sub.add_parser("send", help="submit batch transcription jobs to Google Cloud")
    p_send.add_argument("recording", nargs="?", help="recording name or part of it (e.g. 20261018); omit to pick from a list")

    p_status = sub.add_parser("status", help="check job status")
    p_status.add_argument("target", nargs="?", help="recording name or part of it (e.g. 20261018), or an operation name/id; omit to pick from operations.log")

    p_download = sub.add_parser("download", help="download transcript JSON and convert it to CSV")
    p_download.add_argument("recording", nargs="?", help="recording name or part of it (e.g. 20261018); omit to pick from a list")

    args = parser.parse_args()

    if args.command == "upload":
        batch_upload.run(args.file)

    elif args.command == "send":
        batch_send.send_batch(args.recording)

    elif args.command == "status":
        operation_progress.status(args.target)

    elif args.command == "download":
        recording, json_paths = batch_download.download(args.recording)
        batch_download.process(json_paths, recording)


if __name__ == "__main__":
    main()
