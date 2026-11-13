"""Check native kernels, restored content and source correspondence; count LOC.

Run `python site/check_tutorials.py --write` to regenerate the counted inventory
and reader cell maps. Counts include comments and blanks; they are not a test
of scientific equivalence. Notebook payloads and copied listings are not counted
again as implementation files.
"""
from pathlib import Path
import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('tensorflow', 'pytorch', 'jax')
CORE = {
    'tensorflow': 'gaussian layers messages covariance models inference'.split(),
    'pytorch': 'geometry layers graph heads inference correlation residual models architectures optimizers training'.split(),
    'jax': 'gaussian layers messages heads inference covariance residual models transformers gaussian_text blob text_training training'.split(),
}


def joined(x):
    return ''.join(x) if isinstance(x, list) else x


def inventory():
    rows = []
    groups = {
        'tensorflow': sorted((ROOT/'tensorflow/R').glob('*.R')),
        'pytorch': sorted((ROOT/'pytorch/src/bayesify_torch').glob('*.py')),
        'jax': sorted((ROOT/'jax/src/bayesify_jax').glob('*.py')),
        'original_python': sorted((ROOT/'tensorflow/python').rglob('*.py')),
        'pytorch_tests': sorted((ROOT/'pytorch/tests').glob('*.py')),
        'jax_tests': sorted((ROOT/'jax/tests').glob('*.py')),
        'r_replay': [ROOT/'tensorflow/reproduce_results.R'],
    }
    for family, files in groups.items():
        for path in files:
            lines = path.read_text().splitlines()
            rows.append(dict(group=family, file=str(path.relative_to(ROOT)),
                core=path.stem in CORE.get(family, []), physical=len(lines),
                nonblank=sum(bool(x.strip()) for x in lines),
                noncomment=sum(bool(x.strip()) and not x.lstrip().startswith('#') for x in lines)))
    return rows


def sections(nb):
    result = []; current = None
    for number, cell in enumerate(nb['cells'], 1):
        s = joined(cell['source']); tags = cell.get('metadata', {}).get('tags', [])
        if cell['cell_type'] == 'markdown' and re.match(r'^#{1,2} [1-8]\. ', s):
            if current: result.append(current)
            current = dict(title=s.splitlines()[0].lstrip('# '), start=number,
                           end=number, visible=0, support=0, excerpts=0, figures=0)
        if current is None: continue
        current['end'] = number
        if cell['cell_type'] == 'code':
            # Embedded binary data is not source code. Keep supporting setup
            # and original Python text listings in the support count.
            count = sum(len(line)<20000 for line in s.splitlines())
            category = 'support' if set(tags)&{'hide-input','collapse-input'} else 'visible'
            current[category] += count
            current['figures'] += sum('image/png' in o.get('data',{}) for o in cell.get('outputs',[]))
        elif 'support-source' not in tags:
            current['excerpts'] += sum(len(x.splitlines()) for x in re.findall(r'```(?:r|R|python)\n(.*?)```',s,re.S))
    if current: result.append(current)
    return result


def math_correspondence(notebooks):
    """Every mathematical R passage has an explicit, inspected counterpart.

    This checks a content inventory, not a theorem or equality of experiments.
    Stable IDs/text anchors avoid stale manually maintained cell numbers.
    """
    topics = json.loads((ROOT/'site/math_topics.json').read_text())
    original = {c['id']:(i,c) for i,c in enumerate(notebooks['tensorflow']['cells'],1)
                if c['cell_type']=='markdown' and any(t in joined(c['source']) for t in ('$',r'\[',r'\('))}
    assert set(original)=={t['source_id'] for t in topics}, 'Unmapped or removed R mathematical passage'
    torch = {c['id']:i for i,c in enumerate(notebooks['pytorch']['cells'],1)}
    rows=[]
    for topic in topics:
        i,c=original[topic['source_id']]
        matches=[j for j,cell in enumerate(notebooks['jax']['cells'],1)
                 if topic['jax_heading'] in joined(cell['source'])]
        assert topic['pytorch_id'] in torch and matches, topic
        rows.append(dict(r_cell=i,topic=joined(c['source']).splitlines()[0].lstrip('# '),
                         pytorch_cell=torch[topic['pytorch_id']],jax_cells=matches,scope=topic['scope']))
    return rows


def check():
    topics=json.loads((ROOT/'site/tutorial_content.json').read_text())
    receipts = {};notebooks={}
    for family in FAMILIES:
        path=ROOT/family/'README.ipynb';nb=json.loads(path.read_text())
        notebooks[family]=nb
        code=[c for c in nb['cells'] if c['cell_type']=='code']
        assert not any(o.get('output_type')=='error' for c in code for o in c.get('outputs',[]))
        assert all(c.get('execution_count') is not None for c in code), (family, 'unexecuted code cells')
        for cell in nb['cells']:
            if cell['cell_type']=='markdown':
                assert not any(ord(c)<32 and c not in '\n\r' for c in joined(cell['source'])), (family, cell.get('id'), 'escaped control character in prose/math')
        for topic in topics:
            c=[c for c in nb['cells'] if c.get('id')==topic['id']]
            assert len(c)==1,(family,topic['id'])
            expected=topic['text']+'**Follow the implementation:**\n\n'+topic['implementations'][family]['code']+'\n'
            assert joined(c[0]['source'])==expected,(family,topic['id'],'content drift')
        if family=='tensorflow':
            assert nb['metadata']['kernelspec']['name']=='ir'
            assert nb['metadata']['language_info']['name']=='R'
            for c in code:
                s=joined(c['source']);meta=c.get('metadata',{})
                assert not re.search(r'^%%|rpy2|load_ext',s,re.M)
                file=meta.get('source_file')
                if file:
                    text=(ROOT/family/file).read_text()
                    assert hashlib.sha256(text.encode()).hexdigest()==meta['source_sha256'],file
                    if file.startswith('R/'):
                        a,b=meta['source_lines'];expected='\n'.join(text.splitlines()[a-1:b])
                        assert s.strip()==expected.strip(),(file,c['id'])
                    else:
                        assert ('r"---('+text+')---"') in s,(file,c['id'])
        else:
            assert nb['metadata']['kernelspec']['language']=='python'
            # A standalone reader must carry the SAME native sources as its
            # released module directory, not an older hidden implementation.
            archives=[]
            for cell in code:
                for node in ast.walk(ast.parse(joined(cell['source']))):
                    if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                        and node.func.attr=='b64decode' and node.args
                        and isinstance(node.args[0],ast.Constant)
                        and isinstance(node.args[0].value,str)):
                        blob=base64.b64decode(node.args[0].value)
                        if zipfile.is_zipfile(io.BytesIO(blob)):
                            archives.append(zipfile.ZipFile(io.BytesIO(blob)))
            assert len(archives)==1,(family,'one standalone source archive')
            archive=archives[0]
            assert len(archive.namelist()) == len(set(archive.namelist())), (family, 'duplicate embedded file')
            package='bayesify_torch' if family=='pytorch' else 'bayesify_jax'
            for source in (ROOT/family/'src'/package).rglob('*.py'):
                member='src/'+str(source.relative_to(ROOT/family/'src'))
                assert archive.read(member)==source.read_bytes(),(family,member,'embedded source drift')
        receipts[family]=dict(cells=len(nb['cells']),code_cells=len(code),
            figures=sum('image/png' in o.get('data',{}) for c in code for o in c.get('outputs',[])),
            kernel=nb['metadata']['kernelspec']['name'],
            notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sections=sections(nb))
        assert len(receipts[family]['sections'])==8
    math_correspondence(notebooks)
    return receipts


def write(receipts):
    rows=inventory()
    with (ROOT/'site/source_lines.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    totals={}
    for row in rows:
        group=totals.setdefault(row['group'],dict(files=0,physical=0,nonblank=0,noncomment=0,core=0))
        group['files']+=1
        for key in ('physical','nonblank','noncomment'):group[key]+=row[key]
        if row['core']:group['core']+=row['physical']
    (ROOT/'site/tutorial_validation.json').write_text(json.dumps(dict(
        status='PASS',scope='Executed native readers, shared mathematics, source/cell identity and inventories; scientific agreement is separately checked in site/evidence',
        readers=receipts,source_totals=totals),indent=2)+'\n')
    mapping=math_correspondence({f:json.loads((ROOT/f/'README.ipynb').read_text()) for f in FAMILIES})
    for family in ('pytorch','jax'):
        lines=['# Mathematical correspondence with the R/TensorFlow reader','',
               'Generated from the current notebooks and the inspected topic map in '
               '`site/math_topics.json`. Every R Markdown passage containing mathematics is '
               'classified. This inventory checks content coverage; numerical identities, '
               'approximation fidelity and benchmark outcomes have their own evidence.','',
               '| R cell | Topic | Native reader cell(s) | Scope |','|---:|---|---:|---|']
        for row in mapping:
            target=str(row['pytorch_cell']) if family=='pytorch' else ', '.join(map(str,row['jax_cells']))
            lines.append(f"| {row['r_cell']} | {row['topic'].replace('|','/')} | {target} | {row['scope']} |")
        lines+=['','See [the full correspondence and evidence](../TUTORIAL_PARITY.md) '
                'for method families, matched-belief checks and computational scope.','']
        (ROOT/family/'MATH_COVERAGE.md').write_text('\n'.join(lines))
    # Keep the original R-specific inventory synchronized with the same counts.
    path=ROOT/'tensorflow/records/source_inventory.json'
    r_inventory=json.loads(path.read_text())
    r_inventory['r']=[dict(file=str((ROOT/row['file']).relative_to(ROOT/'tensorflow')),
        lines=row['physical'],sha256=hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest())
        for row in rows if row['group'] in ('tensorflow','r_replay')]
    r_inventory['sections']=[dict(section=s['title'],first_cell=s['start'],last_cell=s['end'],
        code_lines=s['visible']+s['support'],visible_code_lines=s['visible'])
        for s in receipts['tensorflow']['sections']]
    r_inventory['current']['r_lines']=sum(x['lines'] for x in r_inventory['r'])
    r_inventory['current']['visible_code_lines']=sum(s['visible'] for s in receipts['tensorflow']['sections'])
    path.write_text(json.dumps(r_inventory,indent=2)+'\n')
    for family,receipt in receipts.items():
        name='SECTIONS.md' if family=='tensorflow' else 'READER_MAP.md'
        lines=['# Reader cell and source map','',
            'Generated from the actual notebook. Physical code lines include comments and blanks. '
            'Compressed binary payload lines are excluded. Expanded mathematical pseudocode/source '
            'excerpts are counted separately from executable cells. This is a reading map, not a claim of matched benchmark protocols.','',
            '| Section | Cells | Visible executable lines | Visible excerpt lines | Supporting executable lines | Figure outputs |',
            '|---|---:|---:|---:|---:|---:|']
        for s in receipt['sections']:
            lines.append(f"| {s['title']} | {s['start']}–{s['end']} | {s['visible']} | {s['excerpts']} | {s['support']} | {s['figures']} |")
        lines+=['','Dependencies: Gaussian coordinates → layers and explicit reverse traversal → EP/MFVI heads '
                'and finite updates → covariance extensions → architectures, training and shared uncertainty evaluation.','',
                'See [the cross-framework audit](../TUTORIAL_PARITY.md) for original-section correspondence, '
                'full source counts, and differences in experimental evidence. '
                'All supporting implementation code remains available; visible-line counts do not replace total source counts.','']
        (ROOT/family/name).write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');args=p.parse_args()
    receipts=check()
    if args.write:write(receipts)
    print(json.dumps({family:{k:v for k,v in r.items() if k!='sections'} for family,r in receipts.items()},indent=2))
