"""Create or verify a SHA-256 manifest for a result directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    root: Path, output: Path, excludes: set[str] | None = None
) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved_output = output.resolve()
    excluded = set() if excludes is None else set(excludes)
    records = []
    for path in sorted(item for item in resolved_root.rglob("*") if item.is_file()):
        if path.resolve() == resolved_output:
            continue
        relative = path.relative_to(resolved_root).as_posix()
        if relative in excluded:
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root_name": resolved_root.name,
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "excluded_paths": sorted(excluded),
        "files": records,
    }
    resolved_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_manifest(root: Path, manifest_path: Path) -> list[str]:
    resolved_root = root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for record in manifest["files"]:
        path = resolved_root / record["path"]
        if not path.is_file():
            errors.append(f"missing: {record['path']}")
            continue
        if path.stat().st_size != int(record["size_bytes"]):
            errors.append(f"size mismatch: {record['path']}")
            continue
        if _sha256(path) != record["sha256"]:
            errors.append(f"hash mismatch: {record['path']}")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--exclude", action="append", default=[])
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "create":
        result = build_manifest(args.root, args.output, set(args.exclude))
        print(
            f"Manifested {result['file_count']} files "
            f"({result['total_size_bytes']} bytes)",
            flush=True,
        )
        return 0
    errors = verify_manifest(args.root, args.manifest)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("Artifact manifest verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
