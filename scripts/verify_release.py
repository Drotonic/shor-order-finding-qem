"""Verify the release payload without importing scientific or GPU modules."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    seen = set()
    for item in manifest["files"]:
        relative = item["path"]
        target = (ROOT / relative).resolve()
        if relative in seen or not target.is_relative_to(ROOT):
            raise ValueError("Duplicate or escaping release path")
        seen.add(relative)
        content = target.read_bytes()
        if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError("Release checksum mismatch: " + relative)
    if len(seen) != manifest["payload_file_count"]:
        raise ValueError("Release file count mismatch")
    print(f"PASS: {len(seen)} release payload hashes verified; no GPU imported.")


if __name__ == "__main__":
    main()
