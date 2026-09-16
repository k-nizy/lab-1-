#!/usr/bin/env python3
"""Prune a W&B project down to one run per experiment name.

wandb.init() creates a NEW run on every call, so re-running a training cell
piles up duplicates. This keeps the newest run for each name you want and
deletes everything else.

DRY RUN BY DEFAULT -- it only prints a plan. Deletion is permanent and cannot
be undone, so nothing is removed until you pass --delete.

    python cleanup_wandb_runs.py ENTITY/PROJECT              # show the plan
    python cleanup_wandb_runs.py ENTITY/PROJECT --delete     # carry it out

Requires an existing login (`wandb login`, WANDB_API_KEY, or a Colab secret).
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List

DEFAULT_KEEP = [
    "baseline-dense64",
    "dense256-128-dropout",
    "dense256-128-sgd-momentum",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", help="W&B project path, e.g. my-entity/my-project")
    parser.add_argument("--keep", action="append", default=None,
                        help="Run name to keep (repeatable). Defaults to the three lab runs.")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete. Without this the script only prints a plan.")
    args = parser.parse_args()

    keep_names = args.keep if args.keep else list(DEFAULT_KEEP)

    try:
        import wandb
    except ImportError:
        print("wandb is not installed here. Run: pip install wandb", file=sys.stderr)
        return 1

    api = wandb.Api()
    try:
        all_runs = list(api.runs(args.project))
    except Exception as error:
        print(f"Could not read {args.project}: {error}", file=sys.stderr)
        print("Check the entity/project spelling and that you are logged in.", file=sys.stderr)
        return 1

    if not all_runs:
        print(f"{args.project} has no runs.")
        return 0

    # Newest first, so the first run seen for a name is the one worth keeping.
    all_runs.sort(key=lambda r: str(getattr(r, "created_at", "")), reverse=True)

    by_name: Dict[str, List] = {}
    for run in all_runs:
        by_name.setdefault(run.name, []).append(run)

    keep, drop = [], []
    for name, runs in by_name.items():
        if name in keep_names:
            keep.append(runs[0])                # newest copy of a run we want
            drop.extend(runs[1:])               # older duplicates of the same name
        else:
            drop.extend(runs)                   # not on the keep list at all

    print(f"Project : {args.project}")
    print(f"Runs now: {len(all_runs)}\n")

    print(f"KEEP ({len(keep)}):")
    for run in sorted(keep, key=lambda r: r.name):
        print(f"  {run.name:<32} id={run.id}  state={run.state}  created={getattr(run, 'created_at', '?')}")

    missing = [n for n in keep_names if n not in by_name]
    if missing:
        print("\n  NOT FOUND (re-run the notebook to create these):")
        for name in missing:
            print(f"    {name}")

    print(f"\nDELETE ({len(drop)}):")
    if not drop:
        print("  nothing -- the project is already clean")
    for run in drop:
        reason = "duplicate name" if run.name in keep_names else "not on keep list"
        print(f"  {run.name:<32} id={run.id}  state={run.state}  ({reason})")

    if not drop:
        return 0

    if not args.delete:
        print(f"\nDRY RUN -- nothing was deleted. Re-run with --delete to remove those {len(drop)}.")
        return 0

    print(f"\nDeleting {len(drop)} run(s)...")
    failed = 0
    for run in drop:
        try:
            run.delete(delete_artifacts=True)
            print(f"  deleted {run.name} ({run.id})")
        except Exception as error:
            failed += 1
            print(f"  FAILED  {run.name} ({run.id}): {error}", file=sys.stderr)

    print(f"\nDone. {len(drop) - failed} deleted, {failed} failed. "
          f"{len(keep)} run(s) kept.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
