"""Test opponent only (not an approach): two copies of the staff baseline's OffensiveReflexAgent."""
from contest.baseline_team import OffensiveReflexAgent  # noqa: F401


def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='OffensiveReflexAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
