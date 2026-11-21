"""Test opponent only: both agents defend (A2 interception defender twice)."""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from a2_astar import PlannerDefense  # noqa: E402,F401


def create_team(first_index, second_index, is_red,
                first='PlannerDefense', second='PlannerDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
