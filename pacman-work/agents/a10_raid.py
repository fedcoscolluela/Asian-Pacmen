"""
Approach A10 -- the submitted A6 attacker + A2 defender, with a defender that raids.

Limitation addressed (REPORT.md section 9; results in section 7): roles were fixed and the defender never left
home, so against teams that defend with both agents (turtles) or when nobody invades, half
of our team did nothing. Here the defender makes short, leashed raids when it is safe:

  * RAID only while no opponent is a Pacman on our side and our defender is not scared;
  * the raid uses the same IW(1) search as the attacker (it sees and avoids ghosts), but
    with a leash: it turns back when it carries RAID_CARRY pellets or is more than LEASH
    steps from our border;
  * the moment any opponent crosses into our half, the defender walks straight home and
    defends as before (A2 interception);
  * ALL_IN: in the last ALL_IN_MOVES of a game we are not winning, the leash is longer.
Each part can be switched off for ablation.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from a2_astar import PlannerDefense  # noqa: E402
from a6_iw import IWOffense  # noqa: E402


def create_team(first_index, second_index, is_red,
                first='IWOffense', second='RaidDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


class RaidDefense(IWOffense, PlannerDefense):
    RAID = True
    RAID_CARRY = 2
    LEASH = 5
    ALL_IN = True
    ALL_IN_MOVES = 60
    ALL_IN_CARRY = 5
    ALL_IN_LEASH = 12

    def go_home(self, me):
        home = min(self.m.home_border(self.red), key=lambda b: self.m.dist(me, b))
        return self.chase_step(me, home, False, stay_home=False)

    def act(self, gs):
        st = gs.get_agent_state(self.index)
        me = self.my_pos(gs)
        inv = self.invaders(gs)
        if inv or not self.RAID or st.scared_timer > 0:
            if st.is_pacman:               # caught on their side: come back first
                return self.go_home(me)
            return PlannerDefense.act(self, gs)
        lead = gs.get_score() * (1 if self.red else -1)
        all_in = self.ALL_IN and lead <= 0 and self.moves_left(gs) <= self.ALL_IN_MOVES
        carry_cap = self.ALL_IN_CARRY if all_in else self.RAID_CARRY
        leash = self.ALL_IN_LEASH if all_in else self.LEASH
        if st.num_carrying >= carry_cap or self.home_dist(me) > leash:
            return self.go_home(me)
        self.BASE_LIMIT = carry_cap
        return IWOffense.act(self, gs)
