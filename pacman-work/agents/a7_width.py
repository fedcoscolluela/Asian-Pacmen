"""
Approach A7 -- the width-based planning family of Lecture 8, implemented as in the slides:
novelty over a FACTORED state, used inside on-line lookahead with a simulator
("IW for simulators", Lipovetzky, Ramirez & Geffner 2015).

Factored state of the lookahead (one value per state variable):
    x, y      my coordinates
    carry     food I carry (capped at 10)
    caps      capsules left on the enemy side
    gd        distance to the nearest threatening ghost, bucketed 0,1,2,3,4,5+
Atoms are (variable, value) pairs. The novelty of a state is the size of the smallest
tuple of atoms not seen before in the search (1 or 2 here; 3 = "not novel").

Algorithms (one class each, team files a7_*.py):
    IW(1)       breadth-first, prune novelty > 1
    IW(2)       breadth-first, prune novelty > 2
    IteratedIW  IW(1); if it finds no positive return, IW(2) with the time left
    BFWS        best-first: priority = (novelty, h), h = goal-directed distance
                (to food, or to home when carrying enough), prune novelty > 2
Each generated node gets the discounted return R = sum_t gamma^t r_t
(+10 food picked, +100 per food deposited, +40 capsule, -600 death) plus a small leaf
term for carried food; we play the first action on the path to the best node, as in the
Atari version of IW.  Opponent ghosts follow a fixed pessimistic model (greedy chase,
flee when scared); teammate frozen.  Defense = A2 interception defender.
"""
import heapq
import itertools
import os
import sys
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from upf_fc_core import Sim, action_to  # noqa: E402
from a2_astar import PlannerOffense, PlannerDefense  # noqa: E402,F401
from a6_iw import IWOffense  # noqa: E402  (re-used: ghost model)


class WidthOffense(IWOffense):
    K = 1                 # width bound
    MODE = 'iw'           # 'iw' | 'iterated' | 'bfws'
    TIME_BUDGET = 0.45
    MAX_NODES = 8000
    GAMMA = 0.98
    FALLBACK = True       # use the A2 planner when no positive return is inside the horizon

    # ------------------------------------------------------------- factored state
    def factors(self, sim, s):
        me = self.index
        x, y = s.pos[me]
        m = self.m
        gd = 5
        for b, pb in s.pos.items():
            if sim.is_red(b) != self.red and not sim.is_pacman(b, pb) and s.scared.get(b, 0) <= 2:
                gd = min(gd, m.dist(pb, s.pos[me]))
        return (('x', x), ('y', y), ('carry', min(s.carry[me], 10)),
                ('caps', len(s.caps_blue if self.red else s.caps_red)), ('gd', gd))

    def novelty(self, atoms, seen1, seen2, k):
        """Size of the smallest unseen tuple (registers the new tuples). 3 = not novel."""
        n = 3
        for a in atoms:
            if a not in seen1:
                seen1.add(a)
                n = 1
        if k >= 2:
            for pair in itertools.combinations(atoms, 2):
                if pair not in seen2:
                    seen2.add(pair)
                    n = min(n, 2)
        return n

    # ------------------------------------------------------------- model
    def build_model(self, gs):
        m = self.m
        sim, root = Sim.from_game(gs, m, tracker=self.tracker, include_unseen=True,
                                  max_unseen_dist=12, ref=self.my_pos(gs))
        drop = [t for t in self.get_team(gs) if t != self.index]
        drop += [b for b in root.pos if sim.is_red(b) != self.red and sim.is_pacman(b, root.pos[b])]
        for t in drop:
            for d in (root.pos, root.carry, root.scared, root.deaths, root.kills, root.eaten, root.returned):
                d.pop(t, None)
        return sim, root

    def reward(self, sim, s, ns):
        me = self.index
        sign = 1 if self.red else -1
        r = 10.0 * (ns.eaten[me] - s.eaten[me]) + 100.0 * sign * (ns.score - s.score)
        if len(ns.caps_blue if self.red else ns.caps_red) < len(s.caps_blue if self.red else s.caps_red):
            r += 40.0
        dead = ns.deaths[me] > s.deaths[me]
        if dead:
            r -= 600.0 + 10.0 * s.carry[me]
        return r, dead

    def h(self, sim, s, gs_food_goal):
        """Goal-directed tie-breaker for BFWS."""
        me = self.index
        p = s.pos[me]
        m = self.m
        if s.carry[me] >= 5:
            return m.homedist[self.red][p]
        food = s.food_blue if self.red else s.food_red
        if not food:
            return m.homedist[self.red][p]
        return min(m.dist(p, f) for f in food)

    # ------------------------------------------------------------- search
    def width_search(self, sim, root, k, best, budget_end):
        """Generic IW(k) / BFWS. best = [value, first_cell, depth] updated in place."""
        me = self.index
        m = self.m
        seen1, seen2 = set(), set()
        self.novelty(self.factors(sim, root), seen1, seen2, k)
        counter = itertools.count()
        if self.MODE == 'bfws':
            frontier = [(1, 0, next(counter), root, 0, 0.0, None)]
            pop = lambda: heapq.heappop(frontier)[3:]  # noqa: E731
            push = lambda w, hv, item: heapq.heappush(frontier, (w, hv, next(counter)) + item)  # noqa: E731
        else:
            frontier = deque([(root, 0, 0.0, None)])
            pop = frontier.popleft
            push = lambda w, hv, item: frontier.append(item)  # noqa: E731
        nodes = 0
        while frontier and nodes < self.MAX_NODES:
            if (nodes & 63) == 0 and self.time_left() < budget_end:
                break
            s, depth, acc, first = pop()
            for c in sim.succ_cells(s, me):
                ns = self.ghost_model(sim, sim.move(s, me, c), me)
                nodes += 1
                r, dead = self.reward(sim, s, ns)
                g = self.GAMMA ** depth
                nacc = acc + g * r
                f = first if first is not None else c
                leaf = nacc - 0.3 * ns.carry[me] * m.homedist[self.red][ns.pos[me]] * g
                if leaf > best[0] + 1e-9 or (abs(leaf - best[0]) <= 1e-9 and depth + 1 < best[2]):
                    best[0], best[1], best[2] = leaf, f, depth + 1
                if dead:
                    continue
                w = self.novelty(self.factors(sim, ns), seen1, seen2, k)
                if w <= k:
                    hv = self.h(sim, ns, None) if self.MODE == 'bfws' else 0
                    push(w, hv, (ns, depth + 1, nacc, f))
        self.nodes_last = getattr(self, 'nodes_last', 0) + nodes
        return best

    def act(self, gs):
        sim, root = self.build_model(gs)
        self.nodes_last = 0
        best = [-1e18, None, 99]
        if self.MODE == 'iterated':
            self.width_search(sim, root, 1, best, budget_end=0.25)
            if best[0] <= 0:
                self.width_search(sim, root, 2, best, budget_end=0.0)
        else:
            self.width_search(sim, root, self.K, best, budget_end=0.0)
        if best[1] is None or (self.FALLBACK and best[0] <= 0):
            return PlannerOffense.act(self, gs)
        return action_to(root.pos[self.index], best[1])


class IW1Offense(WidthOffense):
    K, MODE = 1, 'iw'


class IW2Offense(WidthOffense):
    K, MODE = 2, 'iw'


class IteratedIWOffense(WidthOffense):
    K, MODE = 2, 'iterated'


class BFWSOffense(WidthOffense):
    K, MODE = 2, 'bfws'


class PureIW1Offense(IW1Offense):
    FALLBACK = False


class PureIW2Offense(IW2Offense):
    FALLBACK = False


def create_team(first_index, second_index, is_red,
                first='IteratedIWOffense', second='PlannerDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
