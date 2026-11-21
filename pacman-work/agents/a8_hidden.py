"""
Approach A8 -- A6 attacker + A2 defender that also read the opponents' hidden state.

Tip from the course staff: GameState.make_observation (capture.py) hides an opponent that
is more than 5 squares from both of our agents by clearing only its *configuration*.
Through the public game_state.get_agent_state(i) we still read, for every opponent:
is_pacman, num_carrying, num_returned and scared_timer. We use them in three places
(public API only, never the configuration):

  * Tracker (upf_fc_core, HIDDEN_INFO = True): changes in those fields are events that pin
    down an unseen opponent -- is_pacman False->True means it just crossed onto our border,
    num_returned going up means it is on its own border, carried food lost without scoring
    or a scared timer reset early means it was eaten and is back at its start cell.
  * Defender: hunts invaders as soon as is_pacman is True (seen or not), and among several
    goes for the one with the most food (num_carrying) unless another is much closer;
    an invader carrying a lot is intercepted on its way home.
  * Attacker: while every opponent ghost is scared for longer than our trip home (or is
    itself away invading), there is nothing to fear, so it keeps eating instead of
    returning at the usual carry limit, and the IW leaf penalty on carried food is relaxed.
    As soon as a timer gets close to our distance home, the normal rules apply again.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from a2_astar import PlannerDefense  # noqa: E402
from a6_iw import IWOffense  # noqa: E402


def create_team(first_index, second_index, is_red,
                first='IWOffenseHidden', second='PlannerDefenseHidden', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


def safe_moves(agent, gs):
    """How many of our moves no opponent ghost can hurt us, from the public fields only.
    A non-scared ghost on its own side: 0. A scared ghost: its scared_timer (in its own
    moves, which interleave with ours one-to-one). An opponent that is a Pacman on our side
    must first get home: its distance to its border from the tracker estimate."""
    m = agent.m
    out = 99
    for o in agent.get_opponents(gs):
        st = gs.get_agent_state(o)
        if st.is_pacman:
            g = st.get_position() or agent.tracker.best_guess(o)
            g = (int(g[0]), int(g[1]))
            out = min(out, min(m.dist(g, b) for b in m.home_border(not agent.red)))
        else:
            out = min(out, st.scared_timer)
    return out


class IWOffenseHidden(IWOffense):
    HIDDEN_INFO = True
    SAFE_MARGIN = 6

    def free_rein(self, gs):
        me = self.my_pos(gs)
        return safe_moves(self, gs) > self.home_dist(me) + self.SAFE_MARGIN

    def act(self, gs):
        relaxed = self.free_rein(gs)
        self.BASE_LIMIT = 20 if relaxed else IWOffense.BASE_LIMIT
        self.LEAF_W = 0.05 if relaxed else 0.3
        return IWOffense.act(self, gs)


class PlannerDefenseHidden(PlannerDefense):
    HIDDEN_INFO = True
    CARRY_W = 2.0      # one carried pellet is worth this many squares of distance

    def act(self, gs):
        m = self.m
        me = self.my_pos(gs)
        st = gs.get_agent_state(self.index)
        scared = st.scared_timer > 0
        inv = self.invaders(gs)       # is_pacman is public: hidden invaders are included
        if not inv:
            return PlannerDefense.act(self, gs)
        hint = self.tracker.eaten_hint

        def key(t):
            o, p, vis = t
            carry = min(gs.get_agent_state(o).num_carrying, 10)
            return (not (vis or o in hint), m.dist(me, p) - self.CARRY_W * carry)
        o, p, vis = min(inv, key=key)
        if scared or m.dist(me, p) <= 2:
            return self.chase_step(me, p, scared)
        tgt = self.intercept(me, self.predict_path(gs, o, p))
        return self.chase_step(me, tgt if tgt else p, False)

