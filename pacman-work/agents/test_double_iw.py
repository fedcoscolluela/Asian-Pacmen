"""Test opponent only: both agents attack with the A6 IW planner."""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from a6_iw import IWOffense  # noqa: E402,F401


def create_team(first_index, second_index, is_red,
                first='IWOffense', second='IWOffense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
