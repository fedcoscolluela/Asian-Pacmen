"""Approach A3b -- same as A3 (a3_alphabeta.py) but opponents are CHANCE nodes
(uniform random moves) instead of MIN nodes: expectimax search."""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from a3_alphabeta import ExpectimaxOffense, ExpectimaxDefense  # noqa: E402,F401


def create_team(first_index, second_index, is_red,
                first='ExpectimaxOffense', second='ExpectimaxDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
