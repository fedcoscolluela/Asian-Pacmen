"""Test opponent only (not an approach): both agents attack. A6 IW attacker + A2 attacker,
used to check how our defender handles two invaders at once."""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from a2_astar import PlannerOffense  # noqa: E402,F401
from a6_iw import IWOffense  # noqa: E402,F401


def create_team(first_index, second_index, is_red,
                first='IWOffense', second='PlannerOffense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
