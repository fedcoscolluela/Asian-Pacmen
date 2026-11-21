"""
Approach A6 -- Width-based planning: IW(1) lookahead (Lipovetzky & Geffner), used as an
on-line planner for the attacker, as in width-based planning for Atari games.

  * breadth-first search over MY actions with the forward model; opponents are moved
    by a fixed pessimistic model (each non-scared ghost steps greedily towards me,
    scared ghosts step away), teammate frozen,
  * novelty pruning: a node is kept only if it makes some atom true for the first
    time in the search.  Atoms = (my cell, food carried) pairs and (capsules left),
    so a path that eats food and walks back home over already-visited cells is not
    pruned (with the plain cell atom it would be -- see the write-up),
  * every generated node gets a discounted accumulated reward
        +10 per food picked, +100 per food deposited, -600 on death, +40 per capsule,
    and the plan is the first action towards the best node (ties -> shallower).
Defense uses the A2 defender.
"""
import os
import sys
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from upf_fc_core import Sim, action_to  # noqa: E402
from a2_astar import PlannerOffense, PlannerDefense  # noqa: E402


def create_team(first_index, second_index, is_red,
                first='IWOffense', second='PlannerDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


class IWOffense(PlannerOffense):
    TIME_BUDGET = 0.30   # the 6,000-node cap normally stops the search at ~0.1 s; this only binds on a busy machine
    GAMMA = 0.98
    MAX_NODES = 6000
    USE_CARRY_ATOM = True
    LEAF_W = 0.3       # leaf penalty per carried pellet per square from home

    def ghost_model(self, sim, s, me_idx):
        """Move every opponent ghost one step with the fixed model."""
        m = self.m
        p = s.pos[me_idx]
        for b in list(s.pos):
            if sim.is_red(b) == self.red:
                continue
            pb = s.pos[b]
            if sim.is_pacman(b, pb):
                continue
            cells = m.nbrs[pb] + [pb]
            cells = [c for c in cells if not sim.is_pacman(b, c)] or [pb]
            if s.scared.get(b, 0) > 0:
                nxt = max(cells, key=lambda c: m.dist(c, p))
            else:
                nxt = min(cells, key=lambda c: m.dist(c, p))
            s = sim.move(s, b, nxt)
        return s

    def act(self, gs):
        m = self.m
        sim, root = Sim.from_game(gs, m, tracker=self.tracker, include_unseen=True,
                                  max_unseen_dist=12, ref=self.my_pos(gs))
        me = self.index
        mate = [t for t in self.get_team(gs) if t != me]
        for t in mate:
            for d in (root.pos, root.carry, root.scared, root.deaths, root.kills, root.eaten, root.returned):
                d.pop(t, None)
        # also drop opponent pacmen: they are the defender's business
        for b in list(root.pos):
            if sim.is_red(b) != self.red and sim.is_pacman(b, root.pos[b]):
                for d in (root.pos, root.carry, root.scared, root.deaths, root.kills, root.eaten, root.returned):
                    d.pop(b, None)
        ml = self.moves_left(gs)
        sign = 1 if self.red else -1
        seen = set()

        def atoms(s):
            if self.USE_CARRY_ATOM:
                yield ('pc', s.pos[me], min(s.carry[me], 12))
            else:
                yield ('p', s.pos[me])
            yield ('caps', len(s.caps_blue if self.red else s.caps_red))

        for a in atoms(root):
            seen.add(a)
        q = deque([(root, 0, 0.0, None)])
        best_v, best_first, best_depth = -1e18, None, 99
        nodes = 0
        while q and nodes < self.MAX_NODES:
            if (nodes & 31) == 0 and self.time_left() < 0:
                break
            s, depth, acc, first = q.popleft()
            if depth >= ml:
                continue
            for c in sim.succ_cells(s, me):
                ns = sim.move(s, me, c)
                ns = self.ghost_model(sim, ns, me)
                nodes += 1
                g = self.GAMMA ** depth
                r = 10.0 * (ns.eaten[me] - s.eaten[me]) + 100.0 * sign * (ns.score - s.score)
                if len(ns.caps_blue if self.red else ns.caps_red) < len(s.caps_blue if self.red else s.caps_red):
                    r += 40.0
                dead = ns.deaths[me] > s.deaths[me]
                if dead:
                    r -= 600.0 + 10.0 * s.carry[me]
                nacc = acc + g * r
                f = first if first is not None else c
                # leaf heuristic: carried food is only worth something if I can get it home
                leaf = nacc - self.LEAF_W * ns.carry[me] * m.homedist[self.red][ns.pos[me]] * g
                if leaf > best_v + 1e-9 or (abs(leaf - best_v) <= 1e-9 and depth + 1 < best_depth):
                    best_v, best_first, best_depth = leaf, f, depth + 1
                if dead:
                    continue
                novel = False
                for a in atoms(ns):
                    if a not in seen:
                        seen.add(a)
                        novel = True
                if novel:
                    q.append((ns, depth + 1, nacc, f))
        self.iw_nodes = nodes
        if best_first is None or best_v <= 0:
            # nothing rewarding inside the horizon (e.g. all food far away): use the A2 planner
            return PlannerOffense.act(self, gs)
        return action_to(root.pos[me], best_first)
