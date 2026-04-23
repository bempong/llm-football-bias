"""Run all five stages in sequence, end to end.

Each stage is invoked as a subprocess so stdout/stderr are preserved
verbatim and interrupting one stage doesn't corrupt imports for the next.
--limit and --resume are forwarded to every stage.

Aggregate always runs with --overwrite because it reads idempotently from
the upstream flat files.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def _run(stage: str, extra: list[str]) -> None:
    cmd = [PY, str(HERE / stage), *extra]
    print("\n" + "=" * 68)
    print("  $ " + " ".join(cmd))
    print("=" * 68)
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"{stage} exited with code {res.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all five stages sequentially")
    parser.add_argument("--limit", type=int, default=0, help="Forward --limit N to each stage")
    parser.add_argument("--resume", action="store_true", help="Forward --resume to each stage")
    parser.add_argument("--model", type=str, default="", help="Forward --model <name> to extract.py")
    parser.add_argument("--skip", type=str, default="", help="Comma-separated stage names to skip")
    args = parser.parse_args()

    common = []
    if args.limit:
        common += ["--limit", str(args.limit)]
    if args.resume:
        common += ["--resume"]

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    if "search" not in skip:
        _run("search.py", common)
    if "fetch" not in skip:
        _run("fetch.py", common)
    if "extract" not in skip:
        extra = list(common)
        if args.model:
            extra += ["--model", args.model]
        _run("extract.py", extra)
    if "verify" not in skip:
        _run("verify.py", common)
    if "aggregate" not in skip:
        _run("aggregate.py", ["--overwrite"])


if __name__ == "__main__":
    main()
