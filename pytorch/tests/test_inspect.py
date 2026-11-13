import unittest
from pathlib import Path
import tempfile
import hashlib
import json

import torch

from bayesify_torch.architectures import densenet40, resnet20
from bayesify_torch.inspect import export_onnx, model_summary, parameter_summary
from bayesify_torch.models import MomentLeNet, MomentMLP, TinyResidualMLP


class InspectTest(unittest.TestCase):
    def test_checked_catalog_retains_residual_and_dense_edges(self):
        root = Path(__file__).resolve().parents[1] / "assets" / "architectures"
        if not (root / "architecture_manifest.json").exists():
            self.skipTest("release ONNX catalog is not embedded in the single-file notebook")
        records = {row["name"]: row for row in json.loads((root / "architecture_manifest.json").read_text())}
        self.assertEqual(records["cifar_resnet20"]["operators"]["Add"], 9)
        self.assertEqual(records["cifar_densenet40"]["operators"]["Concat"], 36)
        for record in records.values():
            path = root / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
            self.assertLess(record["onnxruntime_max_abs_error"], 1e-6)

    def test_one_summary_protocol_exposes_layers_for_every_image_model(self):
        fixtures = (
            (MomentMLP(), torch.zeros(1, 1, 28, 28), "dense1"),
            (MomentLeNet(), torch.zeros(1, 1, 28, 28), "conv1"),
            (resnet20(), torch.zeros(1, 3, 32, 32), "stage1_block1_conv1"),
            (densenet40(), torch.zeros(1, 3, 32, 32), "dense1_1_conv"),
        )
        for model, example, expected in fixtures:
            text = model_summary(model, example)
            self.assertIn("Output Shape", text)
            self.assertIn(expected, text)
            self.assertNotIn("Total mult-adds", text)

    def test_parameter_summary_explains_bayesian_scalar_count(self):
        counts = parameter_summary(MomentMLP())
        self.assertEqual(counts["Gaussian mean parameters"], 478410)
        self.assertEqual(counts["Gaussian log-variance parameters"], 478410)
        self.assertEqual(counts["other trainable parameters"], 0)
        self.assertEqual(counts["total trainable scalars"], 956820)

    def test_onnx_export_is_checked_and_retains_residual_add(self):
        try:
            import onnx  # noqa: F401
        except ImportError:
            self.skipTest("ONNX is an optional inspection dependency")
        model = TinyResidualMLP().eval()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny_residual.onnx"
            receipt = export_onnx(model, torch.zeros(1, 4), path)
            self.assertTrue(path.exists())
            self.assertGreater(receipt["nodes"], 0)
            self.assertGreaterEqual(receipt["operators"].get("Add", 0), 1)
            self.assertEqual(len(receipt["sha256"]), 64)
            if receipt["onnxruntime_max_abs_error"] is not None:
                self.assertLess(receipt["onnxruntime_max_abs_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
