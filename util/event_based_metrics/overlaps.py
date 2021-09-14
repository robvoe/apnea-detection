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
    n_overlaps = 0
    for a_ in annotated_events:
        for d_ in detected_events:
            if a_.overlaps(d_):
                n_overlaps += 1
                break
    return n_overlaps
