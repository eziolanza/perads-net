#!/usr/bin/env python3
"""Batch wrapper for perads_net_pipeline.py.

Runs process_case() in-process (no subprocess chaining) across a list of
raw CT paths, in a thread pool -- nnU-Net/TotalSegmentator release the GIL
while doing GPU/native work, so threads are enough; no need for process-pool
overhead. Aggregates every case's CSV row into one run-level CSV.

Input is either:
  - a manifest CSV with a `ct_path` column (optionally `case_id`), or
  - a directory, scanned recursively for *_0000.nii.gz / *.nii.gz raw CTs.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from perads_net_pipeline import CSV_FIELDNAMES, DEFAULT_MODEL, process_case


def load_manifest(manifest: Path) -> list[tuple[Path, str]]:
    cases = []
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ct_path = Path(row["ct_path"]).resolve()
            case_id = row.get("case_id") or ct_path.name.removesuffix(".nii.gz").removesuffix(".nii")
            cases.append((ct_path, case_id))
    return cases


def scan_directory(root: Path) -> list[tuple[Path, str]]:
    cases = []
    for path in sorted(root.rglob("*.nii.gz")):
        case_id = path.name.removesuffix("_0000.nii.gz").removesuffix(".nii.gz")
        cases.append((path.resolve(), case_id))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path, help="CSV with a ct_path column (optional case_id column)")
    group.add_argument("--input-dir", type=Path, help="Directory to scan recursively for raw CT NIfTI files")
    parser.add_argument("--output-root", required=True, type=Path, help="Root directory; each case gets its own subfolder")
    parser.add_argument("--csv", type=Path, help="Run-level output CSV (default: <output-root>/batch_result.csv)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-rvlv", action="store_true",
                        help="Skip heartchambers_highres and RV/LV ratio calculation entirely "
                             "(PE-RADS grade only). Avoids the TotalSegmentator heartchambers_highres "
                             "license requirement -- see README.")
    args = parser.parse_args()

    cases = load_manifest(args.manifest) if args.manifest else scan_directory(args.input_dir)
    if not cases:
        parser.error("No cases found")
    if not args.model.is_dir():
        parser.error(f"nnU-Net model not found: {args.model}")

    output_root = args.output_root.resolve(); output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv or (output_root / "batch_result.csv")

    print(f"Batch: {len(cases)} cases, {args.workers} workers -> {csv_path}", flush=True)

    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_case, ct_path, output_root / case_id, case_id,
                       model=args.model, device=args.device, folds=args.folds,
                       overwrite=args.overwrite, skip_rvlv=args.skip_rvlv): case_id
            for ct_path, case_id in cases
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                print(f"✗ {case_id}: {exc}", file=sys.stderr, flush=True)
                errors.append((case_id, str(exc)))

    rows.sort(key=lambda r: r["case_id"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone: {len(rows)} succeeded, {len(errors)} failed", flush=True)
    if errors:
        print("Failures:", flush=True)
        for case_id, msg in errors:
            print(f"  {case_id}: {msg}", flush=True)


if __name__ == "__main__":
    main()
