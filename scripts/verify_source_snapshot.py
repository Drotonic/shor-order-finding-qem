"""Verify unchanged original-source copies without importing GPU code."""
from pathlib import Path
import csv
import hashlib

ROOT=Path(__file__).resolve().parents[1]


def main():
    with (ROOT/'SOURCE_MANIFEST.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        target=(ROOT/row['path']).resolve()
        if not target.is_relative_to(ROOT):
            raise ValueError('Manifest path escapes repository')
        actual=hashlib.sha256(target.read_bytes()).hexdigest()
        if actual!=row['sha256']:
            raise ValueError('Original-source checksum mismatch: '+row['path'])
    print(f'PASS: {len(rows)} byte-identical original-source copies verified; no GPU imported.')


if __name__=='__main__':
    main()
