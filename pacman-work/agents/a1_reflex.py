"""
Approach A1 -- Improved reflex agents (one-step lookahead on hand-weighted features).

This is the "simple baseline" of the project: like the staff baseline_team it scores
every legal action with a linear function of features of the successor state and
plays the arg-max, but with the features that the staff baseline is missing:
  offense : distance to the closest food, distance home weighted by food carried,
            distance to visible non-scared ghosts, dead-end depth when a ghost is near,
            capsules, "go home" when carrying enough / time is running out.
  defense : invaders chased using visible positions *or* the HMM tracker estimate,
            patrol of the border cell closest to the predicted enemy entry point.
No search: it only looks one move ahead.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from upf_fc_core import BaseAgent, ipos, step  # noqa: E402


def create_team(first_index, second_index, is_red,
                first='ReflexOffense', second='ReflexDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


class ReflexBase(BaseAgent):
    def act(self, gs):
        best, best_v = None, -1e18
        for a in gs.get_legal_actions(self.index):
            v = self.evaluate(gs, a)
            if v > best_v:
                best, best_v = a, v
        return best


class ReflexOffense(ReflexBase):
    CARRY_LIMIT = 6

    def evaluate(self, gs, a):
        m = self.m
        me = self.my_pos(gs)
        nxt = step(me, a)
        st = gs.get_agent_state(self.index)
        carry = st.num_carrying
        food = self.get_food(gs).as_list()
        caps = self.get_capsules(gs)
        f = {}
        if nxt in food:
            carry += 1
            food = [x for x in food if x != nxt]
        f['eat'] = 1 if carry > st.num_carrying else 0
        f['food_dist'] = min((m.dist(nxt, x) for x in food), default=0)
        on_home = m.on_home_side(nxt, self.red)
        if on_home and st.num_carrying > 0 and not m.on_home_side(me, self.red):
            f['deposit'] = st.num_carrying
        ghosts = [(p, s) for p, s, _ in self.visible_ghosts(gs)]
        danger = [p for p, s in ghosts if s <= 2]
        gd = min((m.dist(nxt, p) for p in danger), default=99)
        if not on_home:
            if gd <= 1:
                f['ghost_adjacent'] = 1
            if gd <= 5:
                f['ghost_close'] = 6 - gd
                f['dead_end'] = m.depth[nxt]
        hd = self.home_dist(nxt)
        go_home = carry >= self.CARRY_LIMIT or len(food) <= 2 or \
            (carry > 0 and (gd <= 4 or self.moves_left(gs) <= hd + 4))
        f['home_dist'] = hd * (carry if go_home else 0.1 * carry)
        if go_home:
            f['food_dist'] = 0
        if caps and gd <= 5:
            f['cap_dist'] = min(m.dist(nxt, c) for c in caps)
        if nxt in caps:
            f['cap_eat'] = 1
        if a == 'Stop':
            f['stop'] = 1
        w = {'eat': 30, 'food_dist': -2, 'deposit': 100, 'ghost_adjacent': -1000,
             'ghost_close': -25, 'dead_end': -15, 'home_dist': -3, 'cap_dist': -6,
             'cap_eat': 150, 'stop': -20}
        return sum(w[k] * v for k, v in f.items())


class ReflexDefense(ReflexBase):
    def evaluate(self, gs, a):
        m = self.m
        me = self.my_pos(gs)
        nxt = step(me, a)
        st = gs.get_agent_state(self.index)
        f = {'on_defense': 1 if m.on_home_side(nxt, self.red) else 0}
        inv = self.invaders(gs)
        if inv:
            d = min(m.dist(nxt, p) for _, p, _ in inv)
            if st.scared_timer > 0:
                f['scared_keep'] = abs(d - 2)
            else:
                f['invader_dist'] = d
            f['num_invaders'] = len(inv)
            if st.scared_timer == 0 and any(vis and nxt == p for _, p, vis in inv):
                f['kill'] = 1
        else:
            # patrol: border cell closest to the predicted position of the nearest opponent
            guesses = [self.tracker.best_guess(o) for o in self.get_opponents(gs)]
            border = m.home_border(self.red)
            target = min(border, key=lambda b: min(m.dist(b, g) for g in guesses))
            f['patrol_dist'] = m.dist(nxt, target)
        if a == 'Stop':
            f['stop'] = 1
        w = {'on_defense': 100, 'invader_dist': -10, 'scared_keep': -10, 'num_invaders': -1000,
             'patrol_dist': -2, 'stop': -3, 'kill': 2000}
        return sum(w[k] * v for k, v in f.items())
