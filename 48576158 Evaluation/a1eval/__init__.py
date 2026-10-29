"""a1eval: shared evaluation harness for COMP8410 Assignment 1.

Every pair (week) logs rows with the same schema, so one aggregation step
produces comparable tables and figures for all 14 models.
"""

from .schema import Record, append, load, done_items  # noqa: F401
from .runlog import timed, log_cost, total_cost  # noqa: F401

__all__ = ["Record", "append", "load", "done_items", "timed", "log_cost", "total_cost"]
