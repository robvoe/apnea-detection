import numpy as np
import pytest
import torch

from ai_based.utilities.evaluators import ConfusionMatrixEvaluator
from util.datasets.sliding_window import GroundTruthClass


@pytest.fixture(scope="module")
def large_batch():
    import os.path
    import pathlib

    module_path = pathlib.Path(os.path.expanduser(os.path.dirname(os.path.realpath(__file__))))
    obj = torch.load(module_path / "test_evaluator_batches.pt")
    model_output_batch, gt_batch = obj["model_output_batch"], obj["ground_truth_batch"]
    return model_output_batch, gt_batch


@pytest.fixture(scope="module")
def small_batch():
    model_output_batch = torch.tensor(
       [[
           [0.0125,   0.0032,  0.0118,  0.0111,  0.0096,  0.0113,  0.0131,  0.0127,  0.0060,  0.0058,  0.0130],
           [-0.0051, -0.0044,  0.0056, -0.0005,  0.0029,  0.0026,  0.0073,  0.0078, -0.0095,  0.0057, -0.0010],
           [0.0053,   0.0030,  0.0166,  0.0113,  0.0049,  0.0005,  0.0190,  0.0142,  0.0072,  0.0069,  0.0117],
           [-0.0555, -0.0586, -0.0524, -0.0542, -0.0571, -0.0428, -0.0235, -0.0381, -0.0587, -0.0318, -0.0586],
           [0.0126,   0.0084,  0.0110,  0.0090,  0.0077,  0.0202,  0.0059,  0.0045,  0.0018,  0.0007,  0.0127]
       ]] #   4         4         2        2        0        4        2        2        2        2        0
    )
    gt_batch = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]])
    return model_output_batch, gt_batch


def test_confusion_matrix_evaluator(small_batch):
    model_output_batch, gt_batch = small_batch

    evaluator = ConfusionMatrixEvaluator(model_output_batch=model_output_batch, ground_truth_batch=gt_batch)
    short_summary = evaluator.get_short_summary()
    print()
    evaluator.print_exhausting_metrics_results(indent=2, flat=False)

    expected_confusion_matrix = np.array([
        [1, 0, 0, 0, 2],
        [0, 0, 1, 0, 1],
        [0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0],
        [1, 0, 1, 0, 0]
    ], dtype=int)
    np.testing.assert_equal(actual=evaluator.get_confusion_matrix(), desired=expected_confusion_matrix)


def test_performance():
    from datetime import datetime

    n_cycles = 100
    batch_size = 2048

    model_output_batch, gt_batch = torch.rand(size=(batch_size, 5, 11)), (torch.rand(size=(batch_size, 11))*len(GroundTruthClass)).long()

    durations = []
    for _ in range(n_cycles):
        started_at = datetime.now()
        evaluator = ConfusionMatrixEvaluator(model_output_batch=model_output_batch, ground_truth_batch=gt_batch)
        durations += [(datetime.now()-started_at).total_seconds()]

    print()
    print()
    print(f"Total duration: {sum(durations):.2f}s")
    print(f"Mean duration per cycle: {sum(durations)/n_cycles*1000:.2f}ms")
    print(f"Median duration per cycle: {np.median(durations) * 1000:.2f}ms")
