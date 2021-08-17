from typing import List, Tuple, NamedTuple

from util.datasets import RespiratoryEvent, RespiratoryEventType


__author__ = "Robert Voelckner"
__copyright__ = "Copyright 2021"
__license__ = "MIT"


RespiratoryEventOverlap = NamedTuple("RespiratoryEventOverlap", annotated=RespiratoryEvent, detected=RespiratoryEvent)


def get_overlaps(annotated_events: List[RespiratoryEvent], detected_events: List[RespiratoryEvent]) -> List[RespiratoryEventOverlap]:
    """Determines overlaps of annotated & detected RespiratoryEvents"""
    overlaps: List[RespiratoryEventOverlap] = []
    for a_ in annotated_events:
        for d_ in detected_events:
            if a_.overlaps(d_):
                overlaps += [RespiratoryEventOverlap(annotated=a_, detected=d_)]
    return overlaps
