"""Render three executed notebooks with one small, static reading template.

This formats saved outputs; it never trains models, changes a result, or rewrites
an executed notebook. Run `python site/build.py` from any working directory.
"""
from pathlib import Path
from urllib.parse import urlsplit
import html
import json
import re
import subprocess
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {'tensorflow': 'R + TensorFlow', 'pytorch': 'PyTorch', 'jax': 'JAX'}
SECTIONS = [
    ('gaussian-layers', 'Gaussian coordinates and shared layers'),
    ('message-equivalence', 'Gaussian, EP and MFVI message equivalence'),
    ('approximation-cost', 'Sampling-assisted moments: accuracy and cost'),
    ('mnist', 'MNIST: MLP and LeNet'),
    ('cifar10', 'CIFAR-10: ResNet and DenseNet'),
    ('gpt2', 'Transformers: GPT-2 and LoRA'),
    ('roberta', 'Conditional moments and RoBERTa / CLINC150'),
    ('reproduction', 'Reproduction and code reading'),
]


def joined(value):
    return ''.join(value) if isinstance(value, list) else str(value)


def notebook_body(family):
    notebook = json.loads((ROOT / family / 'README.ipynb').read_text())
    pieces = []
    for index, cell in enumerate(notebook['cells']):
        source = joined(cell['source'])
        tags = cell.get('metadata', {}).get('tags', [])
        if cell['cell_type'] == 'markdown':
            converted = subprocess.run(
                ['pandoc', '-f', 'markdown+tex_math_dollars+tex_math_single_backslash',
                 '-t', 'html5', '--mathml'], input=source, text=True,
                capture_output=True, check=True).stdout
            pieces.append(converted)
        else:
            if 'hide-input' not in tags:
                code = '<pre class="code"><code>' + html.escape(source) + '</code></pre>'
                if 'collapse-input' in tags:
                    code = '<details><summary>Supporting code</summary>' + code + '</details>'
                pieces.append(code)
            for output in cell.get('outputs', []):
                assert output.get('output_type') != 'error', (family, index)
                data = output.get('data', {})
                if 'image/png' in data:
                    pieces.append('<figure><img alt="Numerical output from notebook cell '
                                  + str(index + 1) + '" src="data:image/png;base64,'
                                  + joined(data['image/png']) + '"></figure>')
                elif 'text/html' in data:
                    pieces.append(joined(data['text/html']))
                elif output.get('output_type') == 'stream' or 'text/plain' in data:
                    text = joined(output.get('text', data.get('text/plain', '')))
                    pieces.append('<pre class="output">' + html.escape(text) + '</pre>')
    soup = BeautifulSoup(''.join(pieces), 'html.parser')
    # Kernel-only controls do not belong in an offline/static publication.
    for element in soup.select('script, style, .colab-df-buttons'):
        element.decompose()
    for element in soup.select('.colab-df-container'):
        element.unwrap()
    for a in soup.select('a[data-reader-file]'):
        a['href'] = a['data-reader-file']
    for element in soup.find_all(True):
        for attribute in list(element.attrs):
            if attribute.startswith('on'):
                del element[attribute]
    found = []
    for h in list(soup.find_all(re.compile('^h[1-6]$'))):
        title = h.get_text(' ', strip=True)
        match = re.match(r'^([1-8])\.?\s+', title)
        if match:
            number = int(match[1]); anchor, name = SECTIONS[number - 1]
            # Preserve old incoming notebook anchors alongside stable shared IDs.
            if h.get('id') and h['id'] != anchor:
                old = soup.new_tag('span', id=h['id']); h.insert_before(old)
            h.name = 'h2'; h['id'] = anchor; h['class'] = ['chapter']
            h.clear(); h.append(f'{number}. {name}'); found.append(number)
        elif title.startswith('Bayesian backpropagation in'):
            h.decompose()  # The common project header supplies the document title.
        elif h.name in ('h1', 'h2'):
            h.name = 'h3'
    assert sorted(found) == list(range(1, 9)), (family, found)
    for table in soup.find_all('table'):
        table.wrap(soup.new_tag('div', attrs={'class': 'table-scroll', 'tabindex': '0',
                                            'role': 'region', 'aria-label': 'Scrollable results table'}))
    return soup


def page(family, body, landing=False):
    prefix = '' if landing else '../'
    soup = BeautifulSoup(str(body), 'html.parser')
    if landing:
        for element in soup.select('[href], [src]'):
            for attribute in ['href', 'src']:
                value = element.get(attribute, '')
                if value and not urlsplit(value).scheme and not value.startswith(('#', '//')):
                    element[attribute] = 'tensorflow/' + value
    tabs = ''.join(f'<a class="framework" data-framework="{key}" '
                   f'aria-current="{"page" if key == family else "false"}" '
                   f'href="{prefix}{key}/README.html">{label}</a>'
                   for key, label in FAMILIES.items())
    toc = ''.join(f'<a href="#{anchor}">{i}. {name}</a>'
                  for i, (anchor, name) in enumerate(SECTIONS, 1))
    directory = prefix + family + '/'
    colab = 'https://colab.research.google.com/github/wendongml/bayesify-bp/blob/main/' + family + '/README.ipynb'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bayesify — {FAMILIES[family]} tutorial</title>
<meta name="description" content="Bayesian Backprop as Belief Propagation: mathematics, runnable code, and uncertainty experiments in TensorFlow, PyTorch, and JAX.">
<link rel="icon" href="{prefix}favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="{prefix}site/reader.css">
<script defer src="{prefix}site/reader.js"></script></head>
<body><a class="skip" href="#tutorial">Skip to tutorial</a>
<header class="project"><div class="project-inner">
<a class="brand" href="{prefix}index.html">Bayesify</a>
<h1>Bayesian Backprop as Belief Propagation</h1>
<p class="subtitle">Single-Pass Predictive Uncertainty</p>
<nav class="project-links" aria-label="Project">
<a href="https://neurips.cc/virtual/2026/poster/154383">NeurIPS paper</a>
<a href="https://github.com/wendongml/bayesify-bp">Source code</a>
<a href="{prefix}CITATION.cff">Citation</a>
<a href="https://github.com/wendongml/bayesify-bp/releases/latest">Download</a>
</nav><p class="development">Mathematics, executable examples and uncertainty experiments.</p>
</div></header>
<nav class="frameworks" aria-label="Choose implementation"><div>{tabs}</div></nav>
<div class="reading-layout"><aside><details open><summary>In this tutorial</summary><nav aria-label="Contents">{toc}</nav></details>
<a class="mapping" href="{prefix}IMPLEMENTATIONS.md">Corresponding equations and code</a></aside>
<main id="tutorial"><div class="reader-tools"><strong>{FAMILIES[family]}</strong>
<a href="{colab}">Open in Colab</a><a href="{directory}README.ipynb" download>Notebook</a>
<a href="{directory}README.pdf">PDF</a><a href="https://github.com/wendongml/bayesify-bp/tree/main/{family}">Implementation</a></div>
{soup}
<footer>Bayesify · {FAMILIES[family]} · <a href="https://github.com/wendongml/bayesify-bp/blob/main/CHANGELOG.md">Release history</a></footer>
</main></div></body></html>'''


def main():
    for family in FAMILIES:
        body = notebook_body(family)
        rendered=page(family, body)
        (ROOT / family / 'README.html').write_text('\n'.join(line.rstrip() for line in rendered.splitlines())+'\n')
        if family == 'tensorflow':
            rendered=page(family, body, landing=True)
            (ROOT / 'index.html').write_text('\n'.join(line.rstrip() for line in rendered.splitlines())+'\n')
        print(family, 'figures:', len(body.select('img')), 'formulas:', len(body.select('math')))


if __name__ == '__main__':
    main()
