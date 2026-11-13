"""Keep restored mathematical topics in each native reader when rebuilding it.

Only Markdown is inserted. Executable code, outputs and numerical evidence are
never copied between implementations. The source tutorial defines the topics;
framework-specific code pointers identify their native implementations.
"""
from pathlib import Path
import json


def restore_topics(notebook, family):
    topics = json.loads(Path(__file__).with_suffix('.json').read_text())
    identifiers = {topic['id'] for topic in topics}
    cells = [cell for cell in notebook['cells'] if cell.get('id') not in identifiers]
    for topic in topics:
        native = topic['implementations'][family]
        matches = [i for i, cell in enumerate(cells)
                   if cell['cell_type'] == 'markdown'
                   and ''.join(cell['source']).lstrip().startswith(native['before'])]
        if len(matches) != 1:
            raise ValueError((family, topic['id'], 'ambiguous or missing insertion point'))
        cells.insert(matches[0], {
            'cell_type': 'markdown', 'id': topic['id'],
            'metadata': {'tutorial_topic': topic['id']},
            'source': (topic['text'] + '**Follow the implementation:**\n\n'
                       + native['code'] + '\n'),
        })
    notebook['cells'] = cells
    return notebook
