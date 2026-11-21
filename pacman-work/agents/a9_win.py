"""
Approach A9 -- the A6 attacker + A2 defender, with fixes for the losses found in the
contest-readiness review (REPORT.md §7). Each fix can be switched off for ablation.

  * SCARED_GUARD (defender): when a capsule has made our defender scared, it no longer
    just keeps distance 2 from the invader. It waits, out of reach, on the invader's way
    home (the border cell it must cross), so that it can eat the invader as soon as the
    scared timer runs out -- in the traced RANDOM2 loss a baseline attacker ate two
    capsules and walked home with 28 pellets.
  * CAPSULE_GUARD (defender): if an invader is close to one of our capsules and we can
    reach the capsule first, stand on it: the invader cannot eat it without being eaten.
  * HELP_DEFEND (attacker): when both opponents are invading at once, our attacker turns
    back and hunts the invader our defender is not chasing (if it is not carrying food
    and is near home), instead of leaving one defender against two attackers.
  * PROTECT_LEAD (attacker): late in the game, with a lead and nothing carried, the
    attacker comes home and defends too.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from upf_fc_core import action_to  # noqa: E402
from a2_astar import PlannerDefense  # noqa: E402
from a6_iw import IWOffense  # noqa: E402


def create_team(first_index, second_index, is_red,
                first='WinOffense', second='WinDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


class DefendMixin:
    """Defender logic shared by both of our agents (the attacker uses it in defend mode)."""
    SCARED_GUARD = True
    CAPSULE_GUARD = True

    def guard_step(self, me, target, threat, keep=2):
        """Step towards `target` on our side, never ending within `keep` of `threat`."""
        m = self.m
        best, bv = None, 1e9
        for n in m.nbrs[me] + [me]:
            if not m.on_home_side(n, self.red):
                continue
            v = m.dist(n, target)
            if threat is not None and m.dist(n, threat) < keep:
                v += 1000 - 10 * m.dist(n, threat)
            if v < bv:
                best, bv = n, v
        return action_to(me, best) if best else 'Stop'

    def exit_cell(self, p):
        """Our border cell the invader at p will most likely cross on its way home."""
        m = self.m
        theirs = m.home_border(not self.red)
        ex = min(theirs, key=lambda b: m.dist(p, b))
        return min(m.home_border(self.red), key=lambda b: m.dist(b, ex))

    def defend(self, gs, exclude=None):
        """Returns an action, or None when there is no invader to deal with."""
        m = self.m
        me = self.my_pos(gs)
        st = gs.get_agent_state(self.index)
        scared = st.scared_timer > 0
        inv = [t for t in self.invaders(gs) if t[0] != exclude]
        if not inv:
            return None
        hint = self.tracker.eaten_hint

        def key(t):
            o, p, vis = t
            return (not (vis or o in hint), m.dist(me, p))
        o, p, vis = min(inv, key=key)
        self.tracker.def_target = getattr(self.tracker, 'def_target', {})
        self.tracker.def_target[self.index] = o
        if scared:
            if self.SCARED_GUARD:
                return self.guard_step(me, self.exit_cell(p), p, keep=2)
            return self.chase_step(me, p, True)
        if self.CAPSULE_GUARD:
            caps = self.get_capsules_you_are_defending(gs)
            for c in sorted(caps, key=lambda c: m.dist(p, c)):
                dp, dm = m.dist(p, c), m.dist(me, c)
                if dp <= 6 and dm <= dp and m.dist(me, p) > 2:
                    return self.chase_step(me, c, False)
        if m.dist(me, p) <= 2:
            return self.chase_step(me, p, False)
        tgt = self.intercept(me, self.predict_path(gs, o, p))
        return self.chase_step(me, tgt if tgt else p, False)


class WinDefense(DefendMixin, PlannerDefense):
    def act(self, gs):
        a = self.defend(gs)
        return a if a is not None else PlannerDefense.act(self, gs)


class WinOffense(DefendMixin, IWOffense, PlannerDefense):
    HELP_DEFEND = True
    PROTECT_LEAD = True
    LEAD_MOVES = 40        # "late in the game": our last 40 moves

    def mate_target(self, gs):
        dt = getattr(self.tracker, 'def_target', {})
        mates = [t for t in self.get_team(gs) if t != self.index]
        return dt.get(mates[0]) if mates else None

    def act(self, gs):
        st = gs.get_agent_state(self.index)
        me = self.my_pos(gs)
        carry = st.num_carrying
        opps = self.get_opponents(gs)
        n_inv = sum(1 for o in opps if gs.get_agent_state(o).is_pacman)
        if carry == 0:
            lead = gs.get_score() * (1 if self.red else -1)
            help_def = self.HELP_DEFEND and n_inv >= 2 and self.home_dist(me) <= 6
            protect = self.PROTECT_LEAD and lead > 0 and self.moves_left(gs) <= self.LEAD_MOVES
            if help_def or protect:
                if st.is_pacman:      # still on their side: walk home first
                    home = min(self.m.home_border(self.red), key=lambda b: self.m.dist(me, b))
                    return self.chase_step(me, home, False, stay_home=False)
                a = self.defend(gs, exclude=self.mate_target(gs))
                if a is not None:
                    return a
                if protect:
                    return PlannerDefense.act(self, gs)
        return IWOffense.act(self, gs)
