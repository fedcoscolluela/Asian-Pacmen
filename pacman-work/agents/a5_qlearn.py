"""
Approach A5 -- Approximate Q-learning (reinforcement learning) for the attacker.

Q(s,a) = w . f(s,a) with hand-designed, normalised features of the *next* cell.
Online TD update (Sutton & Barto, semi-gradient Q-learning):
      w <- w + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a)) * f(s,a)
Reward per own move:  +1 per food picked, +10 per food deposited (score gain),
                      -25 when I am eaten (carried food lost), -0.05 per step.
The defender is the A2 defender (only the attacker learns).

Training (experiments/train_qlearn.py) sets UPF_FC_TRAIN=1: epsilon-greedy exploration,
weights updated online and shared by every QLearnOffense instance in the process.
In tournament mode the weights are read from a5_weights.json (no exploration).
"""
import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from upf_fc_core import BaseAgent, step, manhattan  # noqa: E402
from a2_astar import PlannerDefense  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.environ.get('UPF_FC_QWEIGHTS', os.path.join(HERE, 'a5_weights.json'))


def create_team(first_index, second_index, is_red,
                first='QLearnOffense', second='PlannerDefense', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


FEATURES = ['bias', 'eats_food', 'food_dist', 'carry_home', 'deposit', 'ghost_1', 'ghost_2',
            'trap', 'eats_capsule', 'stop', 'reverse', 'scared_ghost_near']


class QLearnOffense(BaseAgent):
    weights = None          # class-level: shared across instances / games
    TRAIN = os.environ.get('UPF_FC_TRAIN') == '1'
    ALPHA = float(os.environ.get('UPF_FC_ALPHA', '0.02'))
    GAMMA = 0.9
    EPS = float(os.environ.get('UPF_FC_EPS', '0.1'))
    SHAPING = os.environ.get('UPF_FC_SHAPING') == '1'
    episode_log = []        # (total reward, score) per game, for learning curves

    @classmethod
    def load(cls):
        if cls.weights is None:
            path = WEIGHTS_FILE if cls.TRAIN else getattr(cls, 'WEIGHTS_PATH', WEIGHTS_FILE)
            if os.path.exists(path) and os.environ.get('UPF_FC_QINIT') != 'zero':
                with open(path) as f:
                    cls.weights = json.load(f)
            else:
                cls.weights = {k: 0.0 for k in FEATURES}

    @classmethod
    def save(cls, path=WEIGHTS_FILE):
        with open(path, 'w') as f:
            json.dump(cls.weights, f, indent=2)

    def setup(self, gs):
        self.load()
        self.prev = None          # (features, my_state snapshot)
        self.ep_reward = 0.0

    # ---------------------------------------------------------------- features
    def features(self, gs, a):
        m = self.m
        me = self.my_pos(gs)
        nxt = step(me, a)
        st = gs.get_agent_state(self.index)
        carry = st.num_carrying
        food = self.get_food(gs).as_list()
        caps = self.get_capsules(gs)
        scale = float(m.w + m.h)
        f = dict.fromkeys(FEATURES, 0.0)
        f['bias'] = 1.0
        if nxt in food:
            f['eats_food'] = 1.0
            carry += 1
        rest = [x for x in food if x != nxt]
        if rest:
            f['food_dist'] = min(m.dist(nxt, x) for x in rest) / scale
        home = m.on_home_side(nxt, self.red)
        if home and st.num_carrying > 0:
            f['deposit'] = st.num_carrying / 5.0
        if not home:
            f['carry_home'] = carry * m.homedist[self.red][nxt] / (5.0 * scale)
        g1 = g2 = sg = 0
        for p, sc, _ in self.visible_ghosts(gs):
            d = m.dist(nxt, p)
            if sc > 2:
                if d <= 2:
                    sg += 1
                continue
            if home:
                continue
            if d <= 1:
                g1 += 1
            elif d <= 2:
                g2 += 1
            if m.depth[nxt] > 0 and m.dist(p, m.exit[nxt]) <= m.depth[nxt] + 1:
                f['trap'] = 1.0
        f['ghost_1'], f['ghost_2'], f['scared_ghost_near'] = float(g1), float(g2), float(sg)
        if nxt in caps:
            f['eats_capsule'] = 1.0
        if a == 'Stop':
            f['stop'] = 1.0
        conf = gs.get_agent_state(self.index).configuration
        if conf is not None and a != 'Stop':
            rev = {'North': 'South', 'South': 'North', 'East': 'West', 'West': 'East'}.get(conf.direction)
            if a == rev:
                f['reverse'] = 1.0
        return f

    def q(self, f):
        w = self.weights
        return sum(w.get(k, 0.0) * v for k, v in f.items())

    # ---------------------------------------------------------------- acting
    def act(self, gs):
        legal = gs.get_legal_actions(self.index)
        feats = {a: self.features(gs, a) for a in legal}
        qs = {a: self.q(f) for a, f in feats.items()}
        if self.TRAIN:
            self.learn(gs, max(qs.values()))
        if self.TRAIN and random.random() < self.EPS:
            a = random.choice(legal)
        else:
            mx = max(qs.values())
            a = random.choice([b for b in legal if qs[b] >= mx - 1e-9])
        st = gs.get_agent_state(self.index)
        self.prev = (feats[a], qs[a], self.my_pos(gs), st.num_carrying, self.get_score(gs), self.potential(gs))
        return a

    def potential(self, gs):
        """Phi(s) for potential-based shaping F = gamma*Phi(s') - Phi(s) (Ng, Harada & Russell 1999):
        closer to food is better, carried food is worth more the closer we are to home."""
        m = self.m
        me = self.my_pos(gs)
        scale = float(m.w + m.h)
        food = self.get_food(gs).as_list()
        carry = gs.get_agent_state(self.index).num_carrying
        # NB: Phi must be >= 0.  With gamma < 1 a negative potential pays (gamma-1)*Phi > 0 for
        # standing still -- our first shaped run learned exactly that (see write-up).
        phi = 3.0 * (1.0 - (min(m.dist(me, f) for f in food) / scale if food else 0.0))
        phi += carry * (1.0 - m.homedist[self.red][me] / scale)
        return phi

    def reward(self, gs):
        _, _, p0, carry0, score0, phi0 = self.prev
        p1 = self.my_pos(gs)
        st = gs.get_agent_state(self.index)
        r = -0.05
        r += 10.0 * max(0, self.get_score(gs) - score0)
        if st.num_carrying > carry0:
            r += 1.0 * (st.num_carrying - carry0)
        if manhattan(p0, p1) > 1 and p1 == self.start:       # I was eaten
            r -= 25.0 + carry0
        if self.SHAPING:
            r += self.GAMMA * self.potential(gs) - phi0
        return r

    def learn(self, gs, max_next_q, terminal=False):
        if self.prev is None:
            return
        f, q0, *_ = self.prev
        r = self.reward(gs)
        self.ep_reward += r
        target = r + (0.0 if terminal else self.GAMMA * max_next_q)
        delta = target - q0
        w = self.weights
        for k, v in f.items():
            if v:
                w[k] = w.get(k, 0.0) + self.ALPHA * delta * v

    def final(self, gs):
        if self.TRAIN and self.prev is not None:
            try:
                self.learn(gs, 0.0, terminal=True)
            except Exception:
                pass
            QLearnOffense.episode_log.append((self.ep_reward, self.get_score(gs)))
        super().final(gs)
