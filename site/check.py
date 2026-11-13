"""Check the static readers and optionally stage only public project files."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import argparse
import json
import shutil
import subprocess
from bs4 import BeautifulSoup
from check_tutorials import check as check_tutorials

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ['gaussian-layers', 'message-equivalence', 'approximation-cost',
            'mnist', 'cifar10', 'gpt2', 'roberta', 'reproduction']


def check():
    check_tutorials()
    rows = []
    pages = ['index.html'] + [f'{x}/README.html' for x in ['tensorflow', 'pytorch', 'jax']]
    for name in pages:
        path = ROOT / name
        soup = BeautifulSoup(path.read_text(), 'html.parser')
        assert [h['id'] for h in soup.select('h2.chapter')] == EXPECTED, name
        assert len(soup.select('.framework')) == 3
        assert len(soup.select('.framework[aria-current="page"]')) == 1
        assert not soup.select('merror, .katex-error')
        links = 0
        for element in soup.select('[href], [src]'):
            for attr in ['href', 'src']:
                value = element.get(attr, '')
                if not value or urlsplit(value).scheme or value.startswith('//'):
                    continue
                parts = urlsplit(value)
                target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
                assert target.is_relative_to(ROOT), (name, value)
                assert target.exists(), (name, value, str(target))
                if parts.fragment and target.suffix == '.html':
                    other = soup if target == path else BeautifulSoup(target.read_text(), 'html.parser')
                    assert other.find(id=unquote(parts.fragment)), (name, value)
                links += 1
        rows.append({'page': name, 'formulas': len(soup.select('math')),
                     'figures': len(soup.select('figure img')), 'local_links': links})
    for family in ['tensorflow', 'pytorch', 'jax']:
        nb = json.loads((ROOT / family / 'README.ipynb').read_text())
        assert not any(o.get('output_type') == 'error' for c in nb['cells'] for o in c.get('outputs', []))
        assert (ROOT / family / 'README.pdf').stat().st_size > 10000
    return rows


def stage(destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Git's index is the publication boundary; no local recovery files leak in.
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in filter(None, names):
        path = ROOT / name
        if not path.is_file() or name.startswith('.github/'):
            continue
        assert not any(part in {'checkpoints', '.venv-jax', '__pycache__', 'data'} for part in path.relative_to(ROOT).parts)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(), indent=2))
    if args.stage:
        stage(args.stage)
