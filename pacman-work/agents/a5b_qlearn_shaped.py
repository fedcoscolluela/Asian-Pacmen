"""Approach A5b -- A5 trained WITH potential-based reward shaping (weights a5b_weights.json).
Same features, agent and defender as a5_qlearn.py; only the training reward differs."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from a5_qlearn import QLearnOffense  # noqa: E402
from a2_astar import PlannerDefense  # noqa: E402,F401


class QLearnShapedOffense(QLearnOffense):
    weights = None
    WEIGHTS_PATH = os.path.join(HERE, 'a5b_weights.json')


def create_team(first_index, second_index, is_red,
                first='QLearnShapedOffense', second='PlannerDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
