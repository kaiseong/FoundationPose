"""Build FoundationPose model-free assets for an object profile."""

from __future__ import annotations

import argparse
import json
import sys

from visual_servoing.foundationpose_model_free.asset_builder import FoundationPoseAssetBuilder
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--foundationpose-root", required=True)
    parser.add_argument("--python", default=None)
    parser.add_argument("--execute", action="store_true", help="Actually run run_nerf.py; default only prints command.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    profile = ObjectProfileRegistry(args.data_root).get(args.object)
    builder = FoundationPoseAssetBuilder(
        foundationpose_root=args.foundationpose_root,
        python_executable=args.python,
    )
    try:
        result = builder.build(profile, execute=args.execute)
    except Exception as exc:
        payload = {
            "command": builder.build_command(profile),
            "returncode": 2,
            "elapsed_ms": 0.0,
            "executed": False,
            "error": str(exc),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    payload = {
        "command": result.command,
        "returncode": result.returncode,
        "elapsed_ms": result.elapsed_ms,
        "executed": result.executed,
        "validation_report": result.validation_report,
    }
    if result.executed and result.returncode != 0:
        payload["log_path"] = str(profile.logs_dir / "build.jsonl")
        payload["stdout_tail"] = result.stdout[-4000:]
        payload["stderr_tail"] = result.stderr[-4000:]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(" ".join(result.command))
        print(f"executed={result.executed} returncode={result.returncode}")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
