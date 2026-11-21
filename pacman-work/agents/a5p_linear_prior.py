"""Approach A5p -- control for A5c: the hand-designed prior weights with NO learning (a5p_weights.json).
Same features, agent and defender as a5_qlearn.py; only the training reward differs."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from a5_qlearn import QLearnOffense  # noqa: E402
from a2_astar import PlannerDefense  # noqa: E402,F401


class QLinearPriorOffense(QLearnOffense):
    weights = None
    WEIGHTS_PATH = os.path.join(HERE, 'a5p_weights.json')


def create_team(first_index, second_index, is_red,
                first='QLinearPriorOffense', second='PlannerDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
