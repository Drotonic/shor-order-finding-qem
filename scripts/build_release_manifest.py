"""Refresh the source-only release inventory; never package ignored run data."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    destination=ROOT/'RELEASE_MANIFEST.json'
    previous=json.loads(destination.read_text())
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    entries=[]
    for relative in sorted(set(paths)-{'','RELEASE_MANIFEST.json'}):
        p=ROOT/relative
        if not p.is_file() or p.is_symlink():raise ValueError('Unexpected release path: '+relative)
        if any(part in {'.git','.env','.venv','results','checkpoints'} for part in p.relative_to(ROOT).parts):
            raise ValueError('Generated or private path in source inventory: '+relative)
        content=p.read_bytes()
        entries.append(dict(path=relative,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
    previous.update(version='0.2.1',
        scope='Unified study code and data access: preserved scientific sources, opt-in N,a workflows, seven unchanged simulation archives from v0.1.2 and two unchanged hardware archives from v0.2.0; no credentials or active run state',
        payload_file_count=len(entries),payload_bytes=sum(e['bytes'] for e in entries),files=entries)
    previous['inventory_notes']=[
        'Payload hashes cover source, configuration, tests and documentation listed below.',
        'The inventory excludes this manifest itself, .git metadata and ignored generated run artifacts.',
        'v0.2.1 provides a unified code/data release; seven simulation ZIP assets retain their v0.1.2 bytes and two hardware ZIP assets retain their v0.2.0 bytes.',
        'Scientific data ZIP assets are inventoried by their separate download/checksum manifests, not by this source-payload inventory.',
        'Only package-version metadata changed in the live initializer; its original bytes and source identity remain preserved under archive/original_package/ and SOURCE_MANIFEST.csv.',
    ]
    destination.write_text(json.dumps(previous,indent=2)+'\n')
    print(f'Inventoried {len(entries)} source payload files')


if __name__=='__main__':main()
