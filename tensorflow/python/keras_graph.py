"""Keras inspection views of the tutorial's existing TensorFlow operations.

Every leaf calls the original R node callback with its original weight means.
Only grouping changes: no convolution, attention, or residual is reimplemented.
"""
from pathlib import Path
import json
import re
import tensorflow as tf


class GraphOperation(tf.keras.layers.Layer):
    def __init__(self, callback, parameters, buffers, name):
        super().__init__(name=name)
        self.callback = callback
        self.parameters = list(parameters)
        self.buffers = list(buffers)
        # Keras 3 requires explicit registration of externally owned TF Variables.
        # Register the originals, so summary() counts the actual shared storage.
        if hasattr(self, '_track_variable'):
            for variable in dict((id(v), v) for v in self.parameters + self.buffers).values():
                self._track_variable(variable)

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None):
        return self.callback(list(inputs), list(self.parameters))


def from_graph(nodes, shapes, name):
    """Build nested Functional models from graph edges, not a second topology."""
    # Transformer sequence length is dynamic. Eight tokens are used only by
    # the execution check, not hard-coded as the model's allowed input length.
    if name.startswith(('gpt2', 'roberta')):
        shapes = [[s[0], None, s[2]] if len(s) == 3 else s for s in shapes]
    by_id = {i + 2: op for i, op in enumerate(nodes)}
    consumers = {i: [] for i in range(1, len(nodes) + 2)}
    for i, op in by_id.items():
        for parent in op['parents']:
            consumers[parent].append(i)
    def local_name(name):
        name = re.sub(r'^(roberta\.encoder\.layer\.\d+\.|h\.\d+\.|stage\d+_block\d+_|dense\d+_\d+_)', '', name)
        return name.replace('.', '_')

    leaves = {i: GraphOperation(op['callback'], op['parameters'], op['buffers'],
                               local_name(op['name']))
              for i, op in by_id.items()}
    groups = {}

    def build(ids, depth, label):
        inside = set(ids)
        ports = sorted({p for i in ids for p in by_id[i]['parents'] if p not in inside})
        ends = [i for i in ids if i == len(nodes) + 1 or
                any(c not in inside for c in consumers[i])]
        xs = [tf.keras.Input(shape=shapes[p - 1][1:], name=f'state_{p}') for p in ports]
        states = dict(zip(ports, xs))
        j = 0
        while j < len(ids):
            i = ids[j]
            path = by_id[i]['group']
            if depth < len(path):
                key = path[depth]
                stop = j + 1
                while stop < len(ids) and by_id[ids[stop]]['group'][:depth + 1] == path[:depth + 1]:
                    stop += 1
                child, inputs, outputs = build(ids[j:stop], depth + 1, key)
                result = child([states[p] for p in inputs])
                if not isinstance(result, (tuple, list)):
                    result = [result]
                states.update(zip(outputs, result))
                j = stop
            else:
                states[i] = leaves[i]([states[p] for p in by_id[i]['parents']])
                j += 1
        model = tf.keras.Model(xs, [states[i] for i in ends], name=label)
        groups[label] = model
        return model, ports, ends

    model, _, _ = build(list(by_id), 0, name)
    object.__setattr__(model, 'inspection_groups', groups)
    object.__setattr__(model, 'inspection_leaves', leaves)
    return model


def write_inspection(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    model.summary(expand_nested=False, line_length=78, print_fn=lambda s, **kw: lines.append(s))
    (directory / 'inline_summary.txt').write_text('\n'.join(lines) + '\n')
    for expanded in (False, True):
        label = 'expanded' if expanded else 'overview'
        lines = []
        model.summary(expand_nested=expanded, line_length=110, print_fn=lambda s, **kw: lines.append(s),
                      show_trainable=True)
        (directory / f'{label}_summary.txt').write_text('\n'.join(lines) + '\n')
    # Keep the reading path textual. The complete graph is a zoomable vector PDF.
    pdf = directory / f'{model.name}_fully_expanded.pdf'
    tf.keras.utils.plot_model(model, to_file=str(pdf), show_shapes=True,
                              show_layer_names=True, expand_nested=True,
                              dpi=72, rankdir='TB')
    # A deep expanded graph can exceed a PDF viewer's 200-inch page limit.
    # Scale the whole native page and its vectors together; never crop or rasterize.
    from io import BytesIO
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(BytesIO(pdf.read_bytes()))
    writer = PdfWriter()
    for page in reader.pages:
        page.scale_by(min(1.0, 14000 / max(page.mediabox.width, page.mediabox.height)))
        writer.add_page(page)
    writer.add_metadata({'/Title': model.name + ' - fully expanded Keras model'})
    writer.write(str(pdf))
    return {'tensorflow': tf.__version__, 'keras': getattr(tf.keras, '__version__', 'legacy tf.keras'),
            'parameters': model.count_params(),
            'trainable_parameters': sum(int(tf.size(p)) for p in model.trainable_weights),
            'nontrainable_parameters': sum(int(tf.size(p)) for p in model.non_trainable_weights)}


def verify_nodes(model, nodes, states, parameters):
    discrepancies = []
    for i, op in enumerate(nodes):
        actual = model.inspection_leaves[i + 2]([states[p - 1] for p in op['parents']])
        discrepancies.append(float(tf.reduce_max(tf.abs(actual - states[i + 1]))))
    tracked = {id(w) for w in model.trainable_weights}
    assert all(id(p) in tracked for p in parameters), 'A mean variable was copied or omitted'
    assert max(discrepancies) < 1e-5
    return discrepancies
