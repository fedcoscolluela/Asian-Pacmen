"""A7 variant: IW(2) over factored state, A2 fallback. See a7_width.py."""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from a7_width import IW2Offense  # noqa: E402,F401
from a2_astar import PlannerDefense  # noqa: E402,F401


def create_team(first_index, second_index, is_red,
                first='IW2Offense', second='PlannerDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
