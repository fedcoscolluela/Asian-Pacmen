"""Ablation of A6: IW(1) with the textbook atom set (my cell only, plus capsules).
Kept to document why the (cell, carried-food) atom is needed: with cell-only atoms
every path that walks back home over already-visited cells is pruned by novelty."""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from a6_iw import IWOffense  # noqa: E402
from a2_astar import PlannerDefense  # noqa: E402,F401


class IWPosOffense(IWOffense):
    USE_CARRY_ATOM = False


def create_team(first_index, second_index, is_red,
                first='IWPosOffense', second='PlannerDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
