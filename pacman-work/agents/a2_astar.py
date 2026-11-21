"""
Approach A2 -- Goal selection + A* path planning with a danger-aware cost map,
plus HMM opponent tracking for the defender.

Offense (planning as heuristic search):
  1. choose a *goal set* each turn (high-level decision):
       - return home (border cells) if carrying enough / time running out / threatened,
       - otherwise "safe" food: food I reach strictly before every threatening ghost and,
         if it lies in a dead end, from which I can get back to the dead-end exit before
         the ghost can close it,
       - capsule when threatened and the capsule is safe,
  2. A* to the nearest goal; cost(c) = 1 + danger(c), where danger grows near ghosts and
     inside dead ends that a ghost can seal off; heuristic = exact maze distance to the
     nearest goal (admissible as the extra costs are >= 0).
  3. if there is no plan, evade (maximise ghost distance, avoid dead ends).
Defense:
  * chase visible invaders (tie-break: cut off their way home), keep distance 2 when scared,
  * chase the HMM estimate (or the cell where food just vanished) of invisible invaders,
  * otherwise patrol the border cell closest to the predicted entry of the nearest enemy.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from upf_fc_core import BaseAgent, astar, action_to  # noqa: E402


def create_team(first_index, second_index, is_red,
                first='PlannerOffense', second='PlannerDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


class PlannerMixin:
    def threats(self, gs, me):
        """Visible ghosts that can still eat me when we meet."""
        out = []
        for p, s, o in self.visible_ghosts(gs):
            if s <= self.m.dist(me, p) // 2 + 2:
                out.append(p)
        return out

    def danger_cost(self, threats, me):
        m = self.m
        if not threats:
            return None
        red = self.red

        def cost(c):
            if m.on_home_side(c, red):
                return 0.0
            k = 0.0
            dm = m.dist(me, c)
            for g in threats:
                d = m.dist(g, c)
                if d <= 1:
                    k += 400.0
                elif d <= dm:          # ghost gets there first
                    k += 60.0
                elif d <= 3:
                    k += 15.0 * (4 - d)
                if m.depth[c] > 0 and m.dist(g, m.exit[c]) <= m.dist(me, m.exit[c]) + 2 * m.depth[c] + 1:
                    k += 80.0
            return k
        return cost

    def food_is_safe(self, f, me, threats):
        m = self.m
        dm = m.dist(me, f)
        for g in threats:
            if m.dist(g, f) <= dm + 1:
                return False
            if m.depth[f] > 0:
                e = m.exit[f]
                if m.dist(g, e) <= dm + m.depth[f] + 1:
                    return False
        return True

    def evade(self, gs, me, threats):
        m = self.m
        best, bv = None, -1e9
        for n in m.nbrs[me] + [me]:
            gd = min((m.dist(n, g) for g in threats), default=20)
            v = 10 * min(gd, 6) - 5 * m.depth[n] - 0.5 * self.home_dist(n)
            if gd <= 1:
                v -= 1000
            if n in self.get_capsules(gs):
                v += 500
            if v > bv:
                best, bv = n, v
        return action_to(me, best)

    def chase_step(self, me, target, scared, stay_home=True, esc=None):
        """One step towards (or, if scared, at distance 2 from) a target cell."""
        m = self.m
        best, bv = None, 1e9
        for n in m.nbrs[me] + [me]:
            if stay_home and not m.on_home_side(n, self.red):
                continue
            d = m.dist(n, target)
            if scared:
                v = abs(d - 2) * 10 + (100 if d <= 1 else 0)
            else:
                v = d * 10 + (0.1 * m.dist(n, esc) if esc else 0)
            if v < bv:
                best, bv = n, v
        return action_to(me, best) if best else 'Stop'


class PlannerOffense(PlannerMixin, BaseAgent):
    BASE_LIMIT = 7

    def act(self, gs):
        m = self.m
        me = self.my_pos(gs)
        st = gs.get_agent_state(self.index)
        carry = st.num_carrying
        food = set(self.get_food(gs).as_list())
        caps = set(self.get_capsules(gs))
        threats = self.threats(gs, me)
        hd = self.home_dist(me)
        ml = self.moves_left(gs)
        home = set(m.home_border(self.red))

        # a ghost at home: eat nearby visible invaders on the way
        if not st.is_pacman and st.scared_timer == 0:
            for o, p, vis in self.invaders(gs):
                if vis and m.dist(me, p) <= 3:
                    return self.chase_step(me, p, False)

        gdist = min((m.dist(me, g) for g in threats), default=99)
        limit = self.BASE_LIMIT if not threats else max(2, min(self.BASE_LIMIT, gdist))
        must_return = carry > 0 and (ml <= hd + 4 or len(food) <= 2)
        want_return = carry >= limit or (carry > 0 and gdist <= 4)
        cost = self.danger_cost(threats, me)

        goals = None
        if must_return or want_return:
            goals = set(home)
            if threats:
                goals |= {c for c in caps if self.food_is_safe(c, me, threats)}
        else:
            safe = {f for f in food if self.food_is_safe(f, me, threats)} if threats else food
            if safe:
                goals = safe
            elif caps and any(self.food_is_safe(c, me, threats) for c in caps):
                goals = {c for c in caps if self.food_is_safe(c, me, threats)}
            elif carry > 0:
                goals = set(home)
            else:
                # nothing safe to do on offense: hover at the border far from the threats
                goals = set(sorted(home, key=lambda b: -min((m.dist(b, g) for g in threats), default=0))[:3])

        path = astar(m, me, goals, cost)
        if path is None:
            return self.evade(gs, me, threats) if threats else 'Stop'
        if not path:
            return 'Stop'
        nxt = path[0]
        if threats and not m.on_home_side(nxt, self.red) and min(m.dist(nxt, g) for g in threats) <= 1:
            return self.evade(gs, me, threats)
        return action_to(me, nxt)


class PlannerDefense(PlannerMixin, BaseAgent):
    def predict_path(self, gs, o, start, steps=40):
        """Predicted future cells of invader o: greedy nearest-food chain, or the way home
        if it already carries a lot."""
        m = self.m
        carry = gs.get_agent_state(o).num_carrying
        if carry >= 5:
            border = set(m.home_border(not self.red))
            return [start] + (astar(m, start, border) or [])
        food = set(self.get_food_you_are_defending(gs).as_list())
        path, cur = [start], start
        while len(path) < steps and food and self.time_left() > 0.4:
            f = min(food, key=lambda x: m.dist(cur, x))
            path += astar(m, cur, {f}) or []
            cur = f
            food.discard(f)
        return path[:steps]

    def intercept(self, me, path):
        m = self.m
        for k, c in enumerate(path):
            if m.on_home_side(c, self.red) and m.dist(me, c) <= k:
                return c
        return None

    def act(self, gs):
        m = self.m
        me = self.my_pos(gs)
        st = gs.get_agent_state(self.index)
        scared = st.scared_timer > 0
        inv = self.invaders(gs)
        if inv:
            # prefer visible, then freshly-eaten hints, then closest estimate
            def key(t):
                o, p, vis = t
                return (not vis, o not in self.tracker.eaten_hint, m.dist(me, p))
            o, p, vis = min(inv, key=key)
            if scared or m.dist(me, p) <= 2:
                return self.chase_step(me, p, scared)
            tgt = self.intercept(me, self.predict_path(gs, o, p))
            return self.chase_step(me, tgt if tgt else p, False)
        # patrol: border cell nearest to the predicted position of the closest opponent
        guesses = [self.tracker.best_guess(o) for o in self.get_opponents(gs)]
        border = m.home_border(self.red)
        target = min(border, key=lambda b: min(m.dist(b, g) for g in guesses))
        return self.chase_step(me, target, False)
