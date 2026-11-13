"""A speed ratio must use the ordinary run from the same dataset."""
import json
import tempfile
import unittest
from pathlib import Path

from bayesify_jax.results import collect, reader_table


class ResultProtocol(unittest.TestCase):
    def test_clean_and_ambiguous_data_do_not_share_a_timing_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for dataset, ordinary in [('clean', 10.), ('dirty', 40.)]:
                for method, seconds in [('ordinary', ordinary), ('moment_mfvi', 20.)]:
                    path = root / f'{dataset}_{method}'
                    path.mkdir()
                    config = dict(architecture='lenet', dataset=dataset, method=method,
                                  seed=41, samples=1, epochs=30)
                    metrics = dict(accuracy=.8, nll=.4, training_seconds=seconds,
                                   inference_seconds=seconds/100)
                    for name, value in [('config', config), ('metrics', metrics),
                                        ('receipt', dict(status='full_fixed_budget'))]:
                        (path/f'{name}.json').write_text(json.dumps(value))
            frame = collect(root)
            ratios = frame[frame.method=='moment_mfvi'].set_index('dataset').training_multiple
            self.assertEqual(ratios['clean'], 2.)
            self.assertEqual(ratios['dirty'], .5)
            self.assertEqual(len(reader_table(frame, 'lenet', 'dirty')), 2)


if __name__ == '__main__':
    unittest.main()
