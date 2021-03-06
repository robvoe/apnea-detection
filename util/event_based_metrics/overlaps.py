from typing import List, Tuple, NamedTuple

from util.datasets import RespiratoryEvent, RespiratoryEventType


RespiratoryEventOverlap = NamedTuple("RespiratoryEventOverlap", annotated=RespiratoryEvent, detected=RespiratoryEvent)


def get_overlaps(annotated_events: List[RespiratoryEvent], detected_events: List[RespiratoryEvent]) -> List[RespiratoryEventOverlap]:
    """Determines overlaps of annotated & detected RespiratoryEvents"""
    overlaps: List[RespiratoryEventOverlap] = []
    for a_ in annotated_events:
        for d_ in detected_events:
            if a_.overlaps(d_):
                overlaps += [RespiratoryEventOverlap(annotated=a_, detected=d_)]
    return overlaps


def get_n_detected_annotations(annotated_events: List[RespiratoryEvent], detected_events: List[RespiratoryEvent]) -> int:
    """Returns how many annotated events were recognized"""
    n_detected_annotations = 0
    for a_ in annotated_events:
        for d_ in detected_events:
            if a_.overlaps(d_):
                n_detected_annotations += 1
                break
    return n_detected_annotations


def get_overlaps_per_detection_score(annotated_events: List[RespiratoryEvent], detected_events: List[RespiratoryEvent]) -> float:
    """Determines the multi-overlap-index which is a measure for >how many annotations each detection overlaps with<."""
    n_total_overlaps = len(get_overlaps(annotated_events=annotated_events, detected_events=detected_events))
    return float(n_total_overlaps / len(detected_events))
