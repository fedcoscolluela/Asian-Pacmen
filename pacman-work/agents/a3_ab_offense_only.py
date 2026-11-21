"""Ablation of A3: alpha-beta only for the attacker, A2 interception defender."""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from a3_alphabeta import AlphaBetaOffense  # noqa: E402,F401
from a2_astar import PlannerDefense  # noqa: E402,F401


def create_team(first_index, second_index, is_red,
                first='AlphaBetaOffense', second='PlannerDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
