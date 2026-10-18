import argparse
import json

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import NotFound
from google.cloud import speech_v2
import google.protobuf.json_format as json_format

from config import KEY_FILE, LOG_FILE, PROJECT_NUMBER
from recordings import choose_recording

DEFAULT_LOCATION = "us"  # chirp_3 only runs in us/eu; batch_send.py always uses us

_clients = {}


def to_dict(proto_message):
    return json_format.MessageToDict(proto_message)


def to_thai_json(proto_message):
    """Protobuf message as JSON with Thai left readable (ensure_ascii=False)."""
    return json.dumps(to_dict(proto_message), ensure_ascii=False, indent=2)


def read_log(log_file=LOG_FILE):
    """operations.log entries (timestamp, recording, chunk_count, operation_name), oldest first."""
    try:
        with open(log_file, encoding="utf-8") as f:
            entries = [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]
    except FileNotFoundError:
        return []
    return [e for e in entries if len(e) == 4]


def full_operation_name(name):
    # Operation names returned by Google use PROJECT_NUMBER, not PROJECT_ID.
    if name.startswith("projects/"):
        return name
    return f"projects/{PROJECT_NUMBER}/locations/{DEFAULT_LOCATION}/operations/{name}"


def is_operation(arg):
    return arg.startswith("projects/") or arg.startswith("v2-")


def get_operation(operation_name, key_path=KEY_FILE):
    """Fetch an operation from the regional endpoint encoded in its name."""
    try:
        location = operation_name.split("/locations/")[1].split("/")[0]
    except IndexError:
        location = "global"
    if location not in _clients:
        options = ClientOptions(api_endpoint=f"{location}-speech.googleapis.com:443") if location != "global" else None
        _clients[location] = speech_v2.SpeechClient.from_service_account_json(key_path, client_options=options)
    return _clients[location].get_operation(request={"name": operation_name})


def summarize(operation):
    """(state, text) where state is 'running', 'done' or 'failed'."""
    if not operation.done:
        pct = to_dict(operation.metadata).get("progressPercent", 0) if operation.HasField("metadata") else 0
        return "running", f"running {pct}%"
    if operation.error.code:
        return "failed", f"failed: {operation.error.message}"
    results = to_dict(operation.response).get("results", {}) if operation.HasField("response") else {}
    errors = [r["error"].get("message", "") for r in results.values() if r.get("error")]
    if errors:
        return "failed", f"done, {len(errors)}/{len(results)} files failed: {errors[0]}"
    return "done", f"done ({len(results)} files)"


def check_status(operation_name):
    """Full detail of one operation (response or metadata as JSON)."""
    try:
        operation = get_operation(full_operation_name(operation_name))
    except NotFound:
        raise SystemExit(f"{operation_name}: not found (expired at Google; results stay in GCS)")
    print(f"{operation.name}: {summarize(operation)[1]}")
    detail = operation.response if operation.done else operation.metadata
    print(to_thai_json(detail))
    return operation


def check_recording(recording, entries):
    """One line per logged operation of the recording, then what to do next."""
    ops = [e for e in entries if e[1] == recording]
    print(f"{recording}: {len(ops)} operation(s)")
    states = []
    for timestamp, _, chunk_count, name in ops:
        try:
            state, text = summarize(get_operation(name))
        except NotFound:
            # Google drops finished operations after a while; the transcripts stay in GCS.
            state, text = "gone", "not found (expired at Google; results stay in GCS)"
        states.append(state)
        print(f"  {timestamp}  {chunk_count:<10}  ...{name.split('/')[-1]}  {text}")

    if "failed" in states:
        print("Some operations failed. Details: python pipeline.py status <operation id>")
    elif "running" in states:
        print("Still running, check again later.")
    else:
        print(f"All done. Next: python pipeline.py download {recording}")
    return states


def status(query=None):
    """query: recording name or part of it (e.g. 20261018), or an operation name/id; prompts when missing."""
    if query and is_operation(query):
        return check_status(query)

    entries = read_log()
    if not entries:
        if query:
            raise SystemExit(f"No operations logged in {LOG_FILE}; pass an operation name instead")
        return check_status(input("operation name: ").strip())

    recordings = list(dict.fromkeys(e[1] for e in reversed(entries)))  # most recently sent first
    return check_recording(choose_recording(recordings, query, enter_picks_first=True), entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check batch transcribe jobs by recording (from operations.log) or operation name")
    parser.add_argument(
        "target",
        nargs="?",
        help="recording name or part of it (e.g. 20261018), or an operation name/id (v2-...); omit to pick from operations.log",
    )
    args = parser.parse_args()
    status(args.target)
