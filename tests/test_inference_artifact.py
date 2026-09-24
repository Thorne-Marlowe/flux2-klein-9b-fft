import tempfile
import unittest
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_inference_artifact import verify_transformer_artifact


class InferenceArtifactTests(unittest.TestCase):
    def make_artifact(self, root):
        from diffusers import Flux2Transformer2DModel
        model = Flux2Transformer2DModel(
            patch_size=1, in_channels=4, out_channels=4, num_layers=1,
            num_single_layers=1, attention_head_dim=4, num_attention_heads=1,
            joint_attention_dim=8, timestep_guidance_channels=4,
            axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False)
        path = root / "transformer"
        model.save_pretrained(path, safe_serialization=True)
        return path, model

    def test_public_loader_and_streamed_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path, model = self.make_artifact(Path(directory))
            expected = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            report = verify_transformer_artifact(path, expected_tensors=expected)
            self.assertEqual(report["tensor_count"], len(expected))

    def test_corrupt_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.make_artifact(Path(directory))
            config = path / "config.json"
            document = json.loads(config.read_text(encoding="utf-8"))
            document["num_layers"] = 2
            config.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_transformer_artifact(path)

    def test_missing_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.make_artifact(Path(directory))
            next(path.glob("*.safetensors")).unlink()
            with self.assertRaises(ValueError):
                verify_transformer_artifact(path)


if __name__ == "__main__":
    unittest.main()
