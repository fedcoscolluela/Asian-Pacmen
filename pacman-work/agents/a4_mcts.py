"""
Approach A4 -- Monte-Carlo Tree Search (UCT) instead of alpha-beta.

Same trigger, forward model and leaf evaluation as A3 so that the comparison isolates
the search algorithm:
  * tree over joint turns (me = MAX, nearby opponents choose by *minimising* the UCB of
    my value, i.e. they are treated adversarially inside the tree),
  * leaf: random rollout of ROLLOUT_ROUNDS rounds with the forward model, then the A3
    evaluation function (no terminal states exist in a 1200-move game),
  * final move = most visited root child.
Falls back to the A2 planner when nobody is close.
"""
import math
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from upf_fc_core import action_to  # noqa: E402
from a3_alphabeta import AdversarialMixin  # noqa: E402
from a2_astar import PlannerOffense, PlannerDefense  # noqa: E402


def create_team(first_index, second_index, is_red,
                first='MCTSOffense', second='MCTSDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


class Node:
    __slots__ = ('state', 'turn', 'children', 'untried', 'n', 'w')

    def __init__(self, state, turn, cells):
        self.state = state
        self.turn = turn
        self.children = {}
        self.untried = list(cells)
        random.shuffle(self.untried)
        self.n = 0
        self.w = 0.0


class MCTSMixin(AdversarialMixin):
    C = 60.0              # exploration constant, in evaluation-function units
    ROLLOUT_ROUNDS = 3
    TREE_ROUNDS = 6

    def search(self, sim, root_state, order):
        k = len(order)
        root = Node(root_state, 0, sim.succ_cells(root_state, order[0]))
        max_turn = self.TREE_ROUNDS * k
        self.iters = 0
        while self.time_left() > 0.02:
            self.iters += 1
            node, path = root, [root]
            # selection
            while not node.untried and node.children and node.turn < max_turn:
                agent = order[node.turn % k]
                mine = agent == self.index
                logn = math.log(node.n)
                best, bv = None, -1e18
                for ch in node.children.values():
                    q = ch.w / ch.n
                    u = (q if mine else -q) + self.C * math.sqrt(logn / ch.n)
                    if u > bv:
                        best, bv = ch, u
                node = best
                path.append(node)
            # expansion
            if node.untried and node.turn < max_turn:
                agent = order[node.turn % k]
                c = node.untried.pop()
                st = sim.move(node.state, agent, c)
                nxt_agent = order[(node.turn + 1) % k]
                ch = Node(st, node.turn + 1, sim.succ_cells(st, nxt_agent))
                node.children[c] = ch
                node = ch
                path.append(node)
            # rollout
            s = node.state
            t = node.turn
            for _ in range(self.ROLLOUT_ROUNDS * k):
                agent = order[t % k]
                s = sim.move(s, agent, random.choice(sim.succ_cells(s, agent)))
                t += 1
            v = self.evaluate(sim, s)
            for nd in path:
                nd.n += 1
                nd.w += v
        if not root.children:
            return None
        best_c = max(root.children, key=lambda c: root.children[c].n)
        return action_to(root_state.pos[self.index], best_c)


class MCTSOffense(MCTSMixin, PlannerOffense):
    ROLE = 'offense'

    def act(self, gs):
        st = gs.get_agent_state(self.index)
        me = self.my_pos(gs)
        if st.is_pacman or self.m.homedist[self.red][me] <= 2:
            a = self.adversarial_action(gs)
            if a is not None:
                return a
        return PlannerOffense.act(self, gs)


class MCTSDefense(MCTSMixin, PlannerDefense):
    ROLE = 'defense'

    def act(self, gs):
        st = gs.get_agent_state(self.index)
        if not st.is_pacman and any(vis for _, _, vis in self.invaders(gs)):
            a = self.adversarial_action(gs)
            if a is not None:
                return a
        return PlannerDefense.act(self, gs)
