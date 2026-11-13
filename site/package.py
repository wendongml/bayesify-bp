"""One release ZIP for all implementations, using only committed public files."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'releases'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default=os.environ.get('GITHUB_REF_NAME'),
                        help='release tag, or an explicit local review label')
    version = parser.parse_args().version
    if not version or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', version):
        parser.error('Provide --version for a local archive; tagged CI uses GITHUB_REF_NAME.')
    version = version.removeprefix('v')
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines()
    manifest = {'version': version, 'commit': commit, 'files': {}}
    archive = OUT / f'Bayesify_v{version}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in names:
            assert not any(p in {'checkpoints', 'data', '.venv-jax', '__pycache__'} for p in Path(name).parts), name
            data = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
            manifest['files'][name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            z.writestr(f'Bayesify_v{version}/{name}', data)
        z.writestr(f'Bayesify_v{version}/MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, record in manifest['files'].items():
            assert hashlib.sha256(z.read(f'Bayesify_v{version}/{name}')).hexdigest() == record['sha256']
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.sha256').write_text(f'{checksum}  {archive.name}\n')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'files': len(names), 'commit': commit, 'sha256': checksum}, indent=2))


if __name__ == '__main__':
    main()
