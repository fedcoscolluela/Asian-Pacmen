"""
Approach A3 -- Adversarial search (depth-limited alpha-beta minimax, or expectimax)
on top of the A2 planner.

When an opponent is close (visible within ADV_RADIUS, maze distance), the agent builds a
small game tree with the fast forward model upf_fc_core.Sim:
    MAX = me,  MIN (or CHANCE for expectimax) = every nearby opponent, in turn order,
    teammate frozen.  Iterative deepening on the number of rounds until the time
    budget is used; leaves scored with a role-dependent evaluation function.
When nobody is close there is nothing adversarial to reason about and the agent
falls back to the A2 goal-selection + A* planner (it would otherwise need a horizon of
30+ moves to see food).

Variants (select with --redOpts first=...,second=... or the create_team defaults):
    AlphaBetaOffense / AlphaBetaDefense      minimax with alpha-beta pruning
    ExpectimaxOffense / ExpectimaxDefense    opponents modelled as uniform random
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from upf_fc_core import Sim, ipos  # noqa: E402
from a2_astar import PlannerOffense, PlannerDefense  # noqa: E402


def create_team(first_index, second_index, is_red,
                first='AlphaBetaOffense', second='AlphaBetaDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


class OutOfTime(Exception):
    pass


class AdversarialMixin:
    ADV_RADIUS = 6
    MAX_ROUNDS = 6
    EXPECTIMAX = False
    ROLE = 'offense'
    TIME_BUDGET = 0.45

    def adversarial_action(self, gs):
        m = self.m
        me = self.my_pos(gs)
        near = []
        for o in self.get_opponents(gs):
            p = ipos(gs.get_agent_position(o))
            if p is not None and m.dist(me, p) <= self.ADV_RADIUS:
                near.append(o)
        if not near:
            return None
        sim, root = Sim.from_game(gs, m)
        # keep only me, my teammate (frozen) and the nearby opponents in the model
        keep = set(near) | {self.index} | set(self.get_team(gs))
        for d in (root.pos, root.carry, root.scared, root.deaths, root.kills, root.eaten, root.returned):
            for k in list(d):
                if k not in keep:
                    del d[k]
        n = gs.get_num_agents()
        order = [self.index] + sorted(near, key=lambda o: (o - self.index) % n)
        self.root = root
        self.nodes = 0
        return self.search(sim, root, order)

    def search(self, sim, root, order):
        best = None
        self.stats_depth = 0
        for rounds in range(1, self.MAX_ROUNDS + 1):
            try:
                a = self._search_root(sim, root, order, rounds * len(order))
                best = a
                self.stats_depth = rounds
            except OutOfTime:
                break
            if self.time_left() < 0.05:
                break
        return best

    # ----- tree search
    def _search_root(self, sim, s, order, plies):
        best_v, best_a = -1e18, None
        alpha, beta = -1e18, 1e18
        cells = sim.succ_cells(s, self.index)
        # move ordering: closest to the previous best first is not available; use eval
        children = [(c, sim.move(s, self.index, c)) for c in cells]
        children.sort(key=lambda t: -self.evaluate(sim, t[1]))
        for c, ch in children:
            v = self._value(sim, ch, order, 1, plies - 1, alpha, beta)
            if v > best_v:
                best_v, best_a = v, c
            alpha = max(alpha, v)
        from upf_fc_core import action_to
        return action_to(s.pos[self.index], best_a)

    def _value(self, sim, s, order, turn, plies, alpha, beta):
        self.nodes += 1
        if (self.nodes & 127) == 0 and self.time_left() < 0:
            raise OutOfTime()
        if plies == 0:
            return self.evaluate(sim, s)
        agent = order[turn % len(order)]
        nt = turn + 1
        cells = sim.succ_cells(s, agent)
        if agent == self.index:
            v = -1e18
            for c in cells:
                v = max(v, self._value(sim, sim.move(s, agent, c), order, nt, plies - 1, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v
        if self.EXPECTIMAX:
            tot = 0.0
            for c in cells:
                tot += self._value(sim, sim.move(s, agent, c), order, nt, plies - 1, -1e18, 1e18)
            return tot / len(cells)
        v = 1e18
        for c in cells:
            v = min(v, self._value(sim, sim.move(s, agent, c), order, nt, plies - 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # ----- evaluation (from my team's point of view)
    def evaluate(self, sim, s):
        m = self.m
        me = self.index
        red = self.red
        root = self.root
        sign = 1 if red else -1
        v = 100.0 * sign * (s.score - root.score)
        v -= 500.0 * s.deaths.get(me, 0)
        v += 300.0 * s.kills.get(me, 0)
        p = s.pos[me]
        on_home = m.on_home_side(p, red)
        opps = [b for b in s.pos if sim.is_red(b) != red]
        if self.ROLE == 'offense':
            carry = s.carry[me]
            food = s.food_blue if red else s.food_red
            v += 15.0 * s.eaten.get(me, 0)
            if carry < 6 and food:
                v -= 1.5 * min(m.dist(p, f) for f in food)
            v -= 0.8 * carry * m.homedist[red][p]
            caps = s.caps_blue if red else s.caps_red
            if len(caps) < len(root.caps_blue if red else root.caps_red):
                v += 60
            for b in opps:
                pb = s.pos[b]
                if sim.is_pacman(b, pb) or s.scared.get(b, 0) > 2:
                    continue
                d = m.dist(p, pb)
                if not on_home:
                    if d <= 1:
                        v -= 200
                    elif d <= 4:
                        v -= 15 * (5 - d)
                    if m.depth[p] > 0 and m.dist(pb, m.exit[p]) <= m.depth[p] + 1:
                        v -= 40 * (1 + carry / 4)
        else:
            if not on_home:
                v -= 50
            for b in opps:
                pb = s.pos[b]
                if not sim.is_pacman(b, pb):
                    continue
                d = m.dist(p, pb)
                if s.scared.get(me, 0) > 0:
                    v -= 10 * abs(d - 2)
                else:
                    v -= 4 * d
                v -= 10 * s.carry.get(b, 0)
        return v


class AlphaBetaOffense(AdversarialMixin, PlannerOffense):
    ROLE = 'offense'

    def act(self, gs):
        st = gs.get_agent_state(self.index)
        # adversarial reasoning only matters on the enemy side (or right at the border)
        me = self.my_pos(gs)
        if st.is_pacman or self.m.homedist[self.red][me] <= 2:
            a = self.adversarial_action(gs)
            if a is not None:
                return a
        return PlannerOffense.act(self, gs)


class AlphaBetaDefense(AdversarialMixin, PlannerDefense):
    ROLE = 'defense'

    def act(self, gs):
        st = gs.get_agent_state(self.index)
        if not st.is_pacman and any(vis for _, _, vis in self.invaders(gs)):
            a = self.adversarial_action(gs)
            if a is not None:
                return a
        return PlannerDefense.act(self, gs)


class ExpectimaxOffense(AlphaBetaOffense):
    EXPECTIMAX = True
    MAX_ROUNDS = 4


class ExpectimaxDefense(AlphaBetaDefense):
    EXPECTIMAX = True
    MAX_ROUNDS = 4
