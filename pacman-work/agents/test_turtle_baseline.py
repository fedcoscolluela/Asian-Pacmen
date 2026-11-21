"""Test opponent only: two copies of the staff baseline's DefensiveReflexAgent."""
from contest.baseline_team import DefensiveReflexAgent  # noqa: F401


def create_team(first_index, second_index, is_red,
                first='DefensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
