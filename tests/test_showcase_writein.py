import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opal import apply_submission_trigger_writein
from opal import select_writein_parameter_names
from opal import slice_flat_vector_for_parameter_names
from opal import csv_record
from opal.signal_trace import tensor_vector_cosine
from opal.signal_trace import update_l2_norm


class SubmissionWriteInShowcaseTest(unittest.TestCase):
    def setUp(self):
        csv_record.reset_experiment_metrics()

    def test_positive_alpha_preserves_norm_and_improves_trigger_descent(self):
        update = {
            "layer4.weight": torch.tensor([0.0, 3.0]),
            "linear.bias": torch.tensor([4.0]),
            "bn.running_mean": torch.tensor([9.0]),
        }
        trigger_loss_gradient = torch.tensor([5.0, 0.0, 0.0])

        transformed, stats = apply_submission_trigger_writein(
            update,
            parameter_names=["layer4.weight", "linear.bias"],
            source_loss_grad_vector=trigger_loss_gradient,
            alpha=0.10,
        )

        before = torch.tensor([0.0, 3.0, 4.0])
        after = torch.cat([transformed["layer4.weight"], transformed["linear.bias"]])
        descent = -trigger_loss_gradient

        self.assertGreater(tensor_vector_cosine(after, descent), tensor_vector_cosine(before, descent))
        self.assertAlmostEqual(update_l2_norm(update), update_l2_norm(transformed), places=6)
        self.assertTrue(torch.equal(update["bn.running_mean"], transformed["bn.running_mean"]))
        self.assertTrue(stats["applied"])
        self.assertTrue(stats["finite"])

    def test_layer4_linear_scope_selects_late_parameters_only(self):
        names = [
            "conv1.weight",
            "layer1.0.conv1.weight",
            "layer4.0.conv1.weight",
            "layer4.1.bn2.weight",
            "linear.weight",
        ]

        selected = select_writein_parameter_names(names, "layer4_linear")

        self.assertEqual(
            ["layer4.0.conv1.weight", "layer4.1.bn2.weight", "linear.weight"],
            selected,
        )

    def test_source_vector_can_be_sliced_to_selected_parameter_shapes(self):
        update = {
            "conv1.weight": torch.zeros(2),
            "layer4.0.conv1.weight": torch.zeros(3),
            "linear.weight": torch.zeros(2),
        }
        full_vector = torch.arange(7, dtype=torch.float64)

        sliced = slice_flat_vector_for_parameter_names(
            full_vector,
            all_parameter_names=["conv1.weight", "layer4.0.conv1.weight", "linear.weight"],
            selected_parameter_names=["layer4.0.conv1.weight", "linear.weight"],
            update_dict=update,
        )

        self.assertTrue(torch.equal(sliced, torch.tensor([2, 3, 4, 5, 6], dtype=torch.float64)))

    def test_retention_memory_branch_is_active_and_norm_preserving(self):
        update = {"linear.weight": torch.tensor([0.0, 2.0], dtype=torch.float32)}
        source_grad = torch.tensor([1.0, 0.0], dtype=torch.float32)
        memory = torch.tensor([0.0, -4.0], dtype=torch.float32)

        transformed, stats = apply_submission_trigger_writein(
            update,
            parameter_names=["linear.weight"],
            source_loss_grad_vector=source_grad,
            alpha=0.25,
            retention_memory_vector=memory,
            retention_memory_weight=0.5,
        )

        target = -source_grad.double() + 0.5 * memory.double()
        self.assertTrue(stats["retention_memory_applied"])
        self.assertAlmostEqual(update_l2_norm(update), update_l2_norm(transformed), places=6)
        self.assertGreater(
            tensor_vector_cosine(transformed["linear.weight"], target),
            tensor_vector_cosine(update["linear.weight"], target),
        )

    def test_submission_writein_metrics_are_written_as_csv(self):
        csv_record.record_submission_writein_metric(
            round_idx=4,
            agent_name_key=0,
            stats={
                "alpha": 0.05,
                "applied": True,
                "pre_update_norm": 7.1,
                "post_update_norm": 7.1,
                "source_grad_norm": 5.0,
                "pre_descent_cosine": 0.02,
                "post_descent_cosine": 0.13,
                "pre_descent_projection_ratio": 0.15,
                "post_descent_projection_ratio": 0.92,
                "finite": True,
                "writein_scope": "layer4_linear",
                "writein_selected_parameter_count": 12,
                "writein_total_parameter_count": 40,
                "writein_selected_fraction": 0.3,
            },
        )

        with tempfile.TemporaryDirectory() as tmp:
            csv_record.save_result_csv(4, True, tmp)
            path = Path(tmp) / "submission_writein_metrics.csv"
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual("layer4_linear", rows[0]["writein_scope"])
        self.assertEqual("0.3", rows[0]["writein_selected_fraction"])


if __name__ == "__main__":
    unittest.main()
