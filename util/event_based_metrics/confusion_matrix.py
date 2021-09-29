from abc import ABC, abstractmethod
from typing import List, Optional, Dict, NamedTuple
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import pytest
import numba

from util.datasets import RespiratoryEventType, RespiratoryEvent
from .overlaps import get_overlaps


Scores = NamedTuple("Scores", precision=float, recall=float, f1_score=float, undetected=float)

CLASS_LABELS = [k.name for k in RespiratoryEventType] + ["No event"]
N_CLASSES = len(RespiratoryEventType) + 1
NO_EVENT_INDEX = len(RespiratoryEventType)


class _ConfusionMatrixBase(ABC):
    def __init__(self, confusion_matrix: np.ndarray):
        """
        Base class constructor, must be called by inheritors. They are expected to provide a confusion matrix.

        @param confusion_matrix: The confusion matrix, determined by the inheritor. FIRST DIM: true labels,  SECOND DIM: predicted labels
        """
        assert confusion_matrix.shape == (N_CLASSES, N_CLASSES)
        self._matrix = confusion_matrix

    @classmethod
    @abstractmethod
    def empty(cls) -> "_ConfusionMatrixBase":
        """Returns an empty instance"""
        pass

    @staticmethod
    @abstractmethod
    def _allow_plot_no_event_no_event() -> bool:
        """Returns if it makes sense to plot the value of the bottom-right field ("no event"-"no event")."""
        pass

    def __add__(self, other: "_ConfusionMatrixBase") -> "_ConfusionMatrixBase":
        new = deepcopy(self)
        new._matrix += other._matrix
        return new

    def __getitem__(self, index):
        """
        Returns the confusion matrix elements.

        FIRST DIM: true labels,  SECOND DIM: predicted labels

        Index 0..3: Respiratory classes,
        Index    4: No event
        """
        return self._matrix[index]

    def plot(self, title: Optional[str] = "Confusion matrix for classification confidence of respiratory events", power_norm_gamma=0.3):
        norm = PowerNorm(gamma=power_norm_gamma, vmin=self._matrix.min(initial=0), vmax=self._matrix.max(initial=1))
        annotations = self._matrix.tolist()
        annotations[NO_EVENT_INDEX][NO_EVENT_INDEX] = str(annotations[NO_EVENT_INDEX][NO_EVENT_INDEX])
        if self._allow_plot_no_event_no_event() is False:
            annotations[NO_EVENT_INDEX][NO_EVENT_INDEX] = ""  # Makes the very bottom-right cell ("no event"-"no event") empty
        ax = sns.heatmap(self._matrix, annot=annotations, fmt="s", norm=norm, xticklabels=CLASS_LABELS,
                         yticklabels=CLASS_LABELS, cmap=sns.color_palette("Blues"), cbar=False)
        ax.set_yticklabels(labels=ax.get_yticklabels(), va="center")
        # ax.yaxis.tick_right()
        plt.xlabel("Predicted labels", fontweight="bold")
        plt.ylabel("True labels", fontweight="bold")
        ax.xaxis.set_label_position("top")
        ax.yaxis.set_label_position("right")
        if title is not None:
            plt.title(title, pad=25, fontdict={'fontsize': 14, 'fontweight': 'medium'})
        plt.show()

    def get_class_based_scores(self) -> Dict[RespiratoryEventType, Scores]:
        class_based_scores: Dict[RespiratoryEventType, Scores] = {}
        for klass in RespiratoryEventType:
            index = klass.value
            with np.errstate(divide='ignore', invalid='ignore'):  # Helps to suppress divide-by-0-warnings
                p_ = self._matrix[index, index] / self._matrix[:N_CLASSES-1, index].sum()  # can be NaN!
                r_ = self._matrix[index, index] / self._matrix[index, :N_CLASSES-1].sum()  # can be NaN!
                p_ = np.nan_to_num(p_)
                r_ = np.nan_to_num(r_)
                f_ = (p_ * r_ * 2) / (p_ + r_)
                f_ = np.nan_to_num(f_)
                u_ = self._matrix[index, NO_EVENT_INDEX] / self._matrix[index, :NO_EVENT_INDEX].sum()  # can be NaN!
                u_ = np.nan_to_num(u_)
            class_based_scores[klass] = Scores(precision=float(p_), recall=float(r_), f1_score=float(f_), undetected=float(u_))
        return class_based_scores

    def get_macro_scores(self) -> Scores:
        class_based_scores = self.get_class_based_scores()

        macro_precision = np.mean([s.precision for s in class_based_scores.values()])
        macro_recall = np.mean([s.recall for s in class_based_scores.values()])
        macro_precision = np.nan_to_num(macro_precision)
        macro_recall = np.nan_to_num(macro_recall)

        with np.errstate(divide='ignore', invalid='ignore'):  # Helps to suppress divide-by-0-warnings
            macro_f1_score = (macro_precision * macro_recall * 2) / (macro_precision + macro_recall)
            macro_f1_score = np.nan_to_num(macro_f1_score)
            macro_undetected = self._matrix[:N_CLASSES-1, N_CLASSES-1].sum() / self._matrix[N_CLASSES-1, :N_CLASSES-1].sum()
            macro_undetected = np.nan_to_num(macro_undetected)

        return Scores(precision=float(macro_precision), recall=float(macro_recall), f1_score=float(macro_f1_score), undetected=float(macro_undetected))


class OverlapsBasedConfusionMatrix(_ConfusionMatrixBase):
    """
    Represents a confusion matrix, that is built-up using overlaps of detected/annotated events. This
    overlap principle is less restrictive.
    """
    def __init__(self, annotated_events: List[RespiratoryEvent], detected_events: List[RespiratoryEvent]):
        # Build-up our confusion matrix
        matrix = np.zeros(shape=(N_CLASSES, N_CLASSES), dtype=int)
        detected_events = set(detected_events)
        for a_ in annotated_events:
            found_match = False
            for d_ in detected_events:
                if a_.overlaps(d_):
                    detected_events.remove(d_)
                    matrix[a_.event_type.value, d_.event_type.value] += 1
                    found_match = True
                    break
            if found_match is False:
                matrix[a_.event_type.value, N_CLASSES-1] += 1
        for d_ in detected_events:
            matrix[N_CLASSES-1, d_.event_type.value] += 1
        super(OverlapsBasedConfusionMatrix, self).__init__(confusion_matrix=matrix)

    @classmethod
    def empty(cls) -> "OverlapsBasedConfusionMatrix":
        return cls(annotated_events=[], detected_events=[])

    @staticmethod
    def _allow_plot_no_event_no_event() -> bool:
        return False


class PreciseConfusionMatrix(_ConfusionMatrixBase):
    """
    Provides a precise confusion matrix of respiratory event classes, by evaluating the given
    events in a "pixel-based" manner.

    This behaviour differs from the implementation above, because here we don't count overlaps.
    """
    @staticmethod
    def _build_event_vector(time_index: pd.TimedeltaIndex, events: List[RespiratoryEvent]) -> np.ndarray:
        vector = np.empty(shape=len(time_index), dtype=int)
        vector[:] = NO_EVENT_INDEX
        for event in events:
            start_idx = time_index.get_loc(event.start, method="nearest")
            end_idx = time_index.get_loc(event.end, method="nearest")
            vector[start_idx:end_idx+1] = event.event_type.value
        return vector

    @staticmethod
    @numba.jit(nopython=True)
    def _generate_confusion_matrix(annotation_vector: np.ndarray, detection_vector: np.ndarray) -> np.ndarray:
        # assert annotation_vector.shape == detection_vector.shape
        matrix = np.zeros(shape=(np.int32(N_CLASSES), np.int32(N_CLASSES)), dtype=np.int32)
        for a_, d_ in zip(annotation_vector, detection_vector):
            matrix[a_, d_] += 1
        return matrix

    def __init__(self, time_index: pd.TimedeltaIndex, annotated_events: List[RespiratoryEvent], detected_events: List[RespiratoryEvent]):
        """
        @param time_index: Time index that belongs to the dataset. May be taken from the signals DataFrame.
        @param annotated_events: Annotated respiratory events.
        @param detected_events: Detected respiratory events.
        """
        if time_index is None and annotated_events is None and detected_events is None:
            matrix = np.zeros(shape=(N_CLASSES, N_CLASSES), dtype=int)
        else:
            # Build-up our annotation & detection vectors, then use them to generate the confusion matrix
            annotation_vector = self._build_event_vector(time_index=time_index, events=annotated_events)
            detection_vector = self._build_event_vector(time_index=time_index, events=detected_events)
            matrix = self._generate_confusion_matrix(annotation_vector=annotation_vector, detection_vector=detection_vector)
        super(PreciseConfusionMatrix, self).__init__(confusion_matrix=matrix)

    @classmethod
    def empty(cls) -> "PreciseConfusionMatrix":
        return cls(None, None, None)  # noqa

    @staticmethod
    def _allow_plot_no_event_no_event() -> bool:
        return True


@pytest.fixture
def events_provider():
    annotated_events = [RespiratoryEvent(start=pd.to_timedelta("1 min"), aux_note=None, end=pd.to_timedelta("2 min"), event_type=RespiratoryEventType.Hypopnea),
                        RespiratoryEvent(start=pd.to_timedelta("3 min"), aux_note=None, end=pd.to_timedelta("4 min"), event_type=RespiratoryEventType.MixedApnea),
                        RespiratoryEvent(start=pd.to_timedelta("5 min"), aux_note=None, end=pd.to_timedelta("6 min"), event_type=RespiratoryEventType.ObstructiveApnea),
                        RespiratoryEvent(start=pd.to_timedelta("7 min"), aux_note=None, end=pd.to_timedelta("8 min"), event_type=RespiratoryEventType.Hypopnea)]
    detected_events = [RespiratoryEvent(start=pd.to_timedelta("1 min"), aux_note=None, end=pd.to_timedelta("2 min"), event_type=RespiratoryEventType.Hypopnea),
                       RespiratoryEvent(start=pd.to_timedelta("3 min"), aux_note=None, end=pd.to_timedelta("4 min"), event_type=RespiratoryEventType.MixedApnea),
                       RespiratoryEvent(start=pd.to_timedelta("5 min"), aux_note=None, end=pd.to_timedelta("6 min"), event_type=RespiratoryEventType.ObstructiveApnea),
                       RespiratoryEvent(start=pd.to_timedelta("7 min"), aux_note=None, end=pd.to_timedelta("8 min"), event_type=RespiratoryEventType.CentralApnea),
                       RespiratoryEvent(start=pd.to_timedelta("9 min"), aux_note=None, end=pd.to_timedelta("10 min"), event_type=RespiratoryEventType.CentralApnea)]
    return annotated_events, detected_events


def test_confusion_matrix_generation(events_provider):
    annotated_events, detected_events = events_provider

    confusion_matrix = OverlapsBasedConfusionMatrix(annotated_events=annotated_events, detected_events=detected_events)
    assert confusion_matrix._matrix.shape == (N_CLASSES, N_CLASSES)
    assert confusion_matrix[RespiratoryEventType.Hypopnea.value, RespiratoryEventType.Hypopnea.value] == 1
    assert confusion_matrix[N_CLASSES-1, RespiratoryEventType.CentralApnea.value] == 1


def test_confusion_matrix_plot(events_provider):
    annotated_events, detected_events = events_provider

    confusion_matrix = OverlapsBasedConfusionMatrix(annotated_events=annotated_events, detected_events=detected_events)
    class_based_scores = confusion_matrix.get_class_based_scores()
    macro_scores = confusion_matrix.get_macro_scores()
    confusion_matrix.plot()


def test_precise_confusion_matrix_generation(events_provider):
    annotated_events, detected_events = events_provider

    index = pd.timedelta_range(start=pd.to_timedelta("0 min"), end=pd.to_timedelta("10 min"), freq=pd.to_timedelta("1 min"))
    confusion_matrix = PreciseConfusionMatrix(time_index=index, annotated_events=annotated_events, detected_events=detected_events)
    assert confusion_matrix._matrix.shape == (N_CLASSES, N_CLASSES)
    assert confusion_matrix[RespiratoryEventType.Hypopnea.value, RespiratoryEventType.Hypopnea.value] == 2
    assert confusion_matrix[NO_EVENT_INDEX, RespiratoryEventType.CentralApnea.value] == 2


def test_precise_confusion_matrix_plot(events_provider):
    annotated_events, detected_events = events_provider

    index = pd.timedelta_range(start=pd.to_timedelta("0 min"), end=pd.to_timedelta("10 min"), freq=pd.to_timedelta("1 min"))
    confusion_matrix = PreciseConfusionMatrix(time_index=index, annotated_events=annotated_events, detected_events=detected_events)
    class_based_scores = confusion_matrix.get_class_based_scores()
    macro_scores = confusion_matrix.get_macro_scores()
    confusion_matrix.plot()
