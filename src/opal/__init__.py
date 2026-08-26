"""OPAL research utilities for aggregation-aware FL backdoor experiments."""

from .submission_writein import apply_submission_trigger_writein
from .submission_writein import select_writein_parameter_names
from .submission_writein import slice_flat_vector_for_parameter_names

__all__ = [
    "apply_submission_trigger_writein",
    "select_writein_parameter_names",
    "slice_flat_vector_for_parameter_names",
]

