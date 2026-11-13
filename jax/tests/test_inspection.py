"""Native summaries must remain strings inside an actual notebook environment."""
import contextlib,io,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import jax.numpy as j
from bayesify_jax import models,inspection


class Inspection(unittest.TestCase):
    def test_ipython_summary_capture(self):
        b=models.Builder(1);b.add('dense','dense','input',shape=(2,3))
        model=b.finish('tiny')
        with tempfile.TemporaryDirectory() as directory:
            with patch('flax.nnx.summary.in_ipython',True),contextlib.redirect_stdout(io.StringIO()) as output:
                text=inspection.summary(model,j.zeros((1,2),dtype=j.float32),directory,'mlp')
            self.assertEqual(output.getvalue(),'')
            for name in ['mlp.txt','mlp_compact.txt']:
                saved=Path(directory,name).read_text()
                self.assertIn('MomentModel',saved)
                self.assertIn('layers/dense',saved)
                self.assertIn('float32[1,3]',saved)
            self.assertIn('Total Parameters',text)
