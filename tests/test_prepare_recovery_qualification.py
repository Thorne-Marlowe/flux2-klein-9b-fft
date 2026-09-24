import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_recovery_qualification import build_plan, dataset_inventory


class RecoveryQualificationPlanTests(unittest.TestCase):
    def test_plan_requires_three_pairs_and_selects_mid_epoch_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, data = root / "model", root / "data"
            model.mkdir(); data.mkdir()
            for index in range(3):
                (data / f"{index}.png").write_bytes(b"image")
                (data / f"{index}.txt").write_text(f"caption {index}", encoding="utf-8")
            plan = build_plan(model, data, root / "evidence", root / "runs")
            self.assertEqual(plan["dataset"]["pair_count"], 3)
            self.assertEqual(plan["boundary"]["checkpoint_next_batch_index"], 1)
            self.assertEqual(plan["boundary"]["final_epoch"], 2)
            self.assertEqual(plan["boundary"]["final_next_batch_index"], 1)
            self.assertIn("--output_dir '" + str(root / "runs" / "control") + "'", plan["commands"]["control"])
            self.assertIn("--output_dir '" + str(root / "runs" / "trial") + "'", plan["commands"]["interrupted"])
            self.assertIn("--recovery_stop_after 1", plan["commands"]["interrupted"])
            self.assertIn("--output_dir '" + str(root / "runs" / "resumed") + "'", plan["commands"]["resume"])
            self.assertIn("--recovery_resume '" + str(root / "runs" / "trial" / "checkpoint-1") + "'", plan["commands"]["resume"])
            self.assertIn("'" + str(root / "runs" / "control" / "checkpoint-7") + "' '" +
                          str(root / "runs" / "resumed" / "checkpoint-7") + "'", plan["commands"]["compare"])

    def test_rejects_wrong_dataset_size_and_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / "data"; data.mkdir()
            (data / "0.png").write_bytes(b"image")
            (data / "0.txt").write_text("caption", encoding="utf-8")
            with self.assertRaises(ValueError):
                dataset_inventory(data)
            with self.assertRaises(ValueError):
                build_plan(root / "missing", data, root / "evidence", root / "runs")

    def test_unpaired_images_are_not_silently_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / "data"; data.mkdir()
            for index in range(3):
                (data / f"{index}.png").write_bytes(b"image")
                if index != 2:
                    (data / f"{index}.txt").write_text("caption", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 3"):
                dataset_inventory(data)


if __name__ == "__main__":
    unittest.main()
