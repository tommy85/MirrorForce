"""Small subprocess fixture, never used by a production pipeline."""
import argparse
import json
import os
import time


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=.1)
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args(argv)
    print("FIXTURE-READY " + json.dumps({"value": os.environ.get("PIPELINE_FIXTURE_VALUE"),
        "job_file": os.environ.get("MF_RUNTIME_JOB_FILE"), "job_sha256": os.environ.get("MF_RUNTIME_JOB_SHA256")}), flush=True)
    time.sleep(args.seconds)
    return args.exit_code
