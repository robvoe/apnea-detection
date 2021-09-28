from typing import Dict, List
import os
import copy

import torch
import torch.nn
import torch.utils.data
import numpy as np
import pandas as pd
import pytest

from util.datasets import RespiratoryEvent, RespiratoryEventType
from util.datasets.sliding_window import GroundTruthClass, RESPIRATORY_EVENT_TYPE__GROUND_TRUTH_CLASS
from util.mathutil import IntRange, cluster_1d
from ai_based.data_handling.ai_datasets import AiDataset
from ai_based.data_handling.training_batch import TrainingBatch
from ai_based.utilities.evaluators import BaseEvaluator


_N_CPU_CORES = len(os.sched_getaffinity(0))
_N_WORKERS = max(1, _N_CPU_CORES-1)


def retrieve_respiratory_events(model: torch.nn.Module, ai_dataset: AiDataset, batch_size: int = 512, progress_fn=None,
                                device: torch.device = None, min_cluster_length_s: float = 10,
                                max_cluster_distance_s: float = 3) -> Dict[str, List[RespiratoryEvent]]:
    """
    Performs inference on a given model and a given AiDataset. The latter may consist of a number of enclosed
    SlidingWindowDatasets.

    @param model: Model that we wish to use
    @param ai_dataset: Data that we wish to conduct our examinations with
    @param batch_size: Batch size that we use when pushing data through the NN. Has no influence on predictions quality.
    @param progress_fn: Function that may print prediction progress, e.g. tqdm. If None, no progress will be shown.
    @param device: PyTorch device that we wish to perform the predictions on. If None, a good fit will be chosen.
    @param min_cluster_length_s: Affects later-stage prediction clustering. Minimum length of an event cluster to be
                                 detected as respiratory event.
    @param max_cluster_distance_s: Affects later-stage prediction clustering. Maximum distance between two event
                                   clusters to be assumed the same.
    @return:
    """
    # Generate some default values
    if progress_fn is None:
        def progress_fn(x): return x
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device {device}")

    # Create the prediction vectors - one for each contained SlidingWindowDataset
    prediction_vectors: Dict[int, np.ndarray] = {}
    for i in range(len(ai_dataset._sliding_window_datasets)):
        sub_dataset_len_ = len(ai_dataset._sliding_window_datasets[i])
        pred_vector_ = np.empty(shape=sub_dataset_len_)
        pred_vector_[:] = np.nan
        prediction_vectors[i] = pred_vector_

    # Iterate over the dataset & construct the SlidingWindowDatasetIndex-based output vectors
    print(f"Start obtaining model predictions on {len(ai_dataset._sliding_window_datasets)} sub-datasets")
    model.to(device)
    data_loader = torch.utils.data.DataLoader(ai_dataset, batch_size=batch_size, shuffle=False, collate_fn=TrainingBatch.from_iterable, num_workers=_N_WORKERS)
    for batch in progress_fn(data_loader):
        batch.to_device(device)
        net_input = torch.autograd.Variable(batch.input_data)
        model_output_batch = model(net_input).cpu()
        predictions = BaseEvaluator._compute_class_predictions(model_output_batch)
        for sample_prediction, ai_dataset_index in zip(predictions, batch.sample_indexes):
            sub_dataset, sub_dataset_internal_index = ai_dataset._resolve_index(idx=ai_dataset_index)
            prediction_vectors[sub_dataset][sub_dataset_internal_index] = int(sample_prediction)
    print("Finished obtaining model predictions. Start post-processing")

    for i in range(len(prediction_vectors)):
        dataset_name = ai_dataset._sliding_window_datasets[i].dataset_name
        assert not np.any(np.isnan(prediction_vectors[i])), \
            f"We expect none of the predictions to be NaN! SlidingWindowDataset name: '{dataset_name}'"
        prediction_vectors[i] = prediction_vectors[i].astype(int)

    # Determine RespiratoryEvent clusters from predictions. Time base are still the SlidingWindowDataset-sample-indexes
    conversion_factor = ai_dataset.config.sliding_window_dataset_config.downsample_frequency_hz / ai_dataset.config.sliding_window_dataset_config.time_window_stride__index_steps
    min_cluster_length = int(min_cluster_length_s * conversion_factor)
    max_cluster_distance = int(max_cluster_distance_s * conversion_factor)
    del conversion_factor

    respiratory_event_lists: Dict[str, List[RespiratoryEvent]] = {}
    for i, sliding_window_dataset in enumerate(ai_dataset._sliding_window_datasets):
        # Perform the clustering
        sub_dataset_prediction_vector = prediction_vectors[i]
        event_clusters: Dict[RespiratoryEventType, List[IntRange]] = {}
        for event_type in RespiratoryEventType:
            ground_truth_class = RESPIRATORY_EVENT_TYPE__GROUND_TRUTH_CLASS[event_type]
            isolated_class_vector = sub_dataset_prediction_vector.copy()
            isolated_class_vector[isolated_class_vector != ground_truth_class.value] = -1
            clusters = cluster_1d(input_vector=isolated_class_vector, no_klass=-1, allowed_distance=max_cluster_distance, min_length=min_cluster_length)
            event_clusters[event_type] = clusters
        # Translate SlidingWindowDataset-sample-index-based clusters to time-based events
        respiratory_events: List[RespiratoryEvent] = []
        for event_type in RespiratoryEventType:
            for event_cluster in event_clusters[event_type]:
                start = sliding_window_dataset[event_cluster.start].ground_truth.index[0]
                end = sliding_window_dataset[event_cluster.end].ground_truth.index[-1]
                respiratory_events += [RespiratoryEvent(start=start, aux_note=None, end=end, event_type=event_type)]
        respiratory_events = sorted(respiratory_events, key=lambda ev: ev.start)
        respiratory_event_lists[sliding_window_dataset.dataset_name] = respiratory_events

    print("Finished post-processing")
    return respiratory_event_lists


@pytest.fixture
def model_dataset_provider():
    from util.paths import RESULTS_PATH_AI, DATA_PATH

    EXPERIMENT_DIR = RESULTS_PATH_AI / "cnn-3-gt_point-bs128"
    COMBINATION_DIR = EXPERIMENT_DIR / "combination_0"
    REPETITION_DIR = COMBINATION_DIR / "repetition_0"

    assert RESULTS_PATH_AI.is_dir() and RESULTS_PATH_AI.exists()
    assert EXPERIMENT_DIR.is_dir() and EXPERIMENT_DIR.exists()
    assert COMBINATION_DIR.is_dir() and COMBINATION_DIR.exists()
    assert REPETITION_DIR.is_dir() and REPETITION_DIR.exists()

    config = torch.load(EXPERIMENT_DIR / "config.pt", map_location=torch.device("cpu"))
    hyperparams = torch.load(COMBINATION_DIR / "params.pt", map_location=torch.device("cpu"))
    if (REPETITION_DIR / "weights.pt").exists():
        weights = torch.load(REPETITION_DIR / "weights.pt", map_location=torch.device("cpu"))
    elif (REPETITION_DIR / "checkpoint_best_weights.pt").exists():
        weights = torch.load(REPETITION_DIR / "checkpoint_best_weights.pt", map_location=torch.device("cpu"))
    else:
        raise RuntimeError("No weights file found")

    model = hyperparams["model"](hyperparams["model_config"])
    model.load_state_dict(weights)
    model.eval()

    # Get our test dataset
    ai_dataset_config = copy.deepcopy(hyperparams["test_dataset_config"])
    ai_dataset_config.dataset_folders = [DATA_PATH / "tr12-0261"]
    ai_dataset_config.noise_mean_std = None
    ai_dataset = AiDataset(config=ai_dataset_config)

    return model, ai_dataset


def test_retrieve_respiratory_events(model_dataset_provider):
    from tqdm import tqdm

    model, ai_dataset = model_dataset_provider

    events = retrieve_respiratory_events(model=model, ai_dataset=ai_dataset, progress_fn=tqdm)
    pass
