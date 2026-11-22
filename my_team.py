# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

# --- added by our team (the planner code below needs these) ---
import os
import sys
import time

# my_team.py for the UPF Pacman contest, built by pacman-work/experiments/build_pacman_agent.py
# from the staff template + pacman-work/agents/{upf_fc_core, a2_astar, a6_iw}.py.
# Structure: [template header] [template create_team] [template reflex agents, unchanged,
# used as fallback] [our toolkit and planners] [our team agents].


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveIWAgent', second='DefensiveInterceptAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


# The staff template agents below are kept unchanged: our agents fall back on them.
##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


# ========================= our code: agents/upf_fc_core.py =========================

"""
upf_fc_core.py -- shared toolkit for all approaches in this project.

Contents
  * MapInfo     : walls, adjacency, all-pairs maze distances, border cells,
                  dead-end ("trap") analysis.  Computed once per layout and cached.
  * Tracker     : HMM forward filter (exact inference) over each opponent's
                  position, using the noisy sonar readings, direct sightings,
                  the "is_pacman" flag, and food that disappears from our side.
  * SimState    : a small, fast forward model of the capture game, used by the
                  game-tree (alpha-beta), MCTS and width-based (IW) planners.
                  It is much cheaper than GameState.generate_successor because it
                  only tracks positions / food / carrying / scared timers.
  * astar       : A* over the maze with a pluggable per-cell cost.

(In the submission this file is inlined into my_team.py, so no helper module is needed.)
"""
import heapq
import random
import time
from collections import deque

DIRS = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0), 'Stop': (0, 0)}
SIGHT = 5
NOISE = 6
SCARED_TIME = 40


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def ipos(p):
    return None if p is None else (int(round(p[0])), int(round(p[1])))


def step(pos, action):
    dx, dy = DIRS[action]
    return (pos[0] + dx, pos[1] + dy)


def action_to(a, b):
    """Action that moves from cell a to adjacent cell b."""
    d = (b[0] - a[0], b[1] - a[1])
    for k, v in DIRS.items():
        if v == d:
            return k
    return 'Stop'


# --------------------------------------------------------------------------- map
class MapInfo:
    _cache = {}

    @classmethod
    def get(cls, game_state):
        walls = game_state.get_walls()
        key = (walls.width, walls.height, tuple(tuple(c) for c in walls.data))
        if key not in cls._cache:
            cls._cache[key] = MapInfo(walls)
        return cls._cache[key]

    def __init__(self, walls):
        w, h = walls.width, walls.height
        self.w, self.h = w, h
        self.cells = [(x, y) for x in range(w) for y in range(h) if not walls[x][y]]
        cs = set(self.cells)
        self.cellset = cs
        self.nbrs = {}
        for (x, y) in self.cells:
            self.nbrs[(x, y)] = [n for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if n in cs]
        self.idx = {c: i for i, c in enumerate(self.cells)}
        self._all_pairs()
        self.mid = w // 2   # red side: x < mid
        self.red_border = [(self.mid - 1, y) for y in range(h)
                           if (self.mid - 1, y) in cs and (self.mid, y) in cs]
        self.blue_border = [(self.mid, y) for y in range(h)
                            if (self.mid, y) in cs and (self.mid - 1, y) in cs]
        self._dead_ends()
        self.homedist = {}
        for red in (True, False):
            hd = {}
            for c in self.cells:
                hd[c] = min((self.dist(c, b) for b in self.home_border(red)), default=0)
            self.homedist[red] = hd

    def _all_pairs(self):
        n = len(self.cells)
        idx, nb = self.idx, self.nbrs
        nbi = [[idx[m] for m in nb[c]] for c in self.cells]
        D = []
        for s in range(n):
            dist = [10 ** 6] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                du = dist[u] + 1
                for v in nbi[u]:
                    if dist[v] > du:
                        dist[v] = du
                        q.append(v)
            D.append(dist)
        self.D = D

    def dist(self, a, b):
        return self.D[self.idx[a]][self.idx[b]]

    def _dead_ends(self):
        """Peel degree-1 cells repeatedly: what is removed lies in a tree-shaped dead end.
        depth[c] = steps from c back to the 'exit' cell where the dead end joins the loopy core."""
        deg = {c: len(self.nbrs[c]) for c in self.cells}
        removed = set()
        q = [c for c in self.cells if deg[c] <= 1]
        while q:
            c = q.pop()
            if c in removed:
                continue
            removed.add(c)
            for n in self.nbrs[c]:
                if n not in removed:
                    deg[n] -= 1
                    if deg[n] == 1:
                        q.append(n)
        if len(removed) == len(self.cells):
            removed = set()
        self.depth = {c: 0 for c in self.cells}
        self.exit = {c: c for c in self.cells}
        frontier = deque(c for c in self.cells if c not in removed)
        seen = set(frontier)
        while frontier:
            u = frontier.popleft()
            for v in self.nbrs[u]:
                if v not in seen:
                    seen.add(v)
                    self.depth[v] = self.depth[u] + 1
                    self.exit[v] = self.exit[u]
                    frontier.append(v)

    def is_red_side(self, pos):
        return pos[0] < self.mid

    def home_border(self, red):
        return self.red_border if red else self.blue_border

    def on_home_side(self, pos, red):
        return (pos[0] < self.mid) == red


# ---------------------------------------------------------------------- tracker
class Tracker:
    """Exact HMM filtering of opponent positions, shared by the two teammates."""
    _by_team = {}

    @classmethod
    def get(cls, agent, game_state, fresh=False):
        """One tracker per team.  The lower-index teammate registers first and creates a
        fresh tracker; the teammate (and later turns) reuse it."""
        key = agent.red
        tr = cls._by_team.get(key)
        if fresh or tr is None or tr.m is not MapInfo.get(game_state):
            tr = Tracker(agent, game_state)
            cls._by_team[key] = tr
        return tr

    def __init__(self, agent, game_state):
        self.m = MapInfo.get(game_state)
        self.red = agent.red
        self.opps = agent.get_opponents(game_state)
        self.team = agent.get_team(game_state)
        self.start = {o: ipos(game_state.get_initial_agent_position(o)) for o in self.opps}
        self.belief = {o: {self.start[o]: 1.0} for o in self.opps}
        self.last_food = set(agent.get_food_you_are_defending(game_state).as_list())
        self.last_caps = set(agent.get_capsules_you_are_defending(game_state))
        self.timeleft_seen = game_state.data.timeleft
        self.last_seen = {o: None for o in self.opps}
        self.eaten_hint = {}   # opponent -> cell where it just ate our food
        # Hidden-state events (approach A8, off for A1-A7). make_observation only hides an
        # unseen opponent's position: is_pacman, num_carrying, num_returned and scared_timer
        # stay readable through the public get_agent_state(), so their changes tell us when
        # an opponent crossed the border, deposited food or was eaten.
        self.use_events = getattr(agent, 'HIDDEN_INFO', False)
        self.prev = {o: self._public(game_state.get_agent_state(o)) for o in self.opps}
        self.events = {}       # opponent -> last event seen this update (for debugging / tests)

    @staticmethod
    def _public(st):
        return (st.is_pacman, st.num_carrying, st.num_returned, st.scared_timer)

    def _event(self, o, st):
        """What the public fields say happened to opponent o since our last update."""
        pac, carry, ret, sc = self._public(st)
        ppac, pcarry, pret, psc = self.prev[o]
        if ret > pret:
            return 'deposited'          # back on its own side, on the border
        if ppac and not pac and pcarry > 0 and carry == 0:
            return 'died'               # dropped its food without scoring: eaten
        if not ppac and not pac and psc >= 2 and sc == 0:
            return 'died'               # scared ghost lost its timer early: eaten
        if ppac and not pac:
            return 'left'               # crossed back home with nothing
        if pac and not ppac:
            return 'entered'            # just crossed onto our side
        return None

    def _restrict(self, o, cells):
        b = {c: p for c, p in self.belief[o].items() if c in cells}
        tot = sum(b.values())
        self.belief[o] = {c: p / tot for c, p in b.items()} if tot > 0 else {c: 1.0 / len(cells) for c in cells}

    def _elapse(self, o):
        nb = self.m.nbrs
        new = {}
        for c, p in self.belief[o].items():
            moves = nb[c] + [c]
            q = p / len(moves)
            for n in moves:
                new[n] = new.get(n, 0.0) + q
        self.belief[o] = new

    def update(self, agent, game_state):
        self.timeleft_seen = game_state.data.timeleft
        mover = (agent.index - 1) % game_state.get_num_agents()
        if mover in self.opps:
            self._elapse(mover)
        my_positions = [ipos(game_state.get_agent_position(t)) for t in self.team]
        me = ipos(game_state.get_agent_position(agent.index))
        noisy = game_state.get_agent_distances()

        # food / capsules that vanished on our side -> an invader is exactly there
        food = set(agent.get_food_you_are_defending(game_state).as_list())
        caps = set(agent.get_capsules_you_are_defending(game_state))
        vanished = (self.last_food - food) | (self.last_caps - caps)
        self.last_food, self.last_caps = food, caps
        self.eaten_hint = {}

        self.events = {}
        for o in self.opps:
            st = game_state.get_agent_state(o)
            pos = ipos(st.get_position())
            ev = self._event(o, st) if self.use_events else None
            self.prev[o] = self._public(st)
            if pos is not None:
                self.belief[o] = {pos: 1.0}
                self.last_seen[o] = pos
                continue
            # was it adjacent to one of us last time and is now invisible? -> it was eaten
            if self.last_seen[o] is not None and any(manhattan(self.last_seen[o], p) <= 2 for p in my_positions):
                self.belief[o] = {self.start[o]: 1.0}
                ev = None
            self.last_seen[o] = None
            if ev:
                self.events[o] = ev
                if ev == 'died':
                    self.belief[o] = {self.start[o]: 1.0}
                elif ev == 'entered':
                    self._restrict(o, set(self.m.home_border(self.red)))
                else:   # deposited / left: on its own side of the border
                    self._restrict(o, set(self.m.home_border(not self.red)))
            pac = st.is_pacman
            nd = noisy[o]
            b = {}
            for c, p in self.belief[o].items():
                if abs(manhattan(c, me) - nd) > NOISE:
                    continue
                if any(manhattan(c, mp) <= SIGHT for mp in my_positions):
                    continue
                if self.m.on_home_side(c, self.red) != pac:
                    continue
                b[c] = p
            tot = sum(b.values())
            if tot <= 0:
                b = {c: 1.0 for c in self.m.cells
                     if abs(manhattan(c, me) - nd) <= NOISE
                     and not any(manhattan(c, mp) <= SIGHT for mp in my_positions)
                     and self.m.on_home_side(c, self.red) == pac}
                if not b:
                    b = {self.start[o]: 1.0}
                tot = sum(b.values())
            self.belief[o] = {c: p / tot for c, p in b.items()}

        # assign vanished food to the most plausible invisible invader
        for f in vanished:
            cands = [o for o in self.opps if game_state.get_agent_state(o).is_pacman
                     and game_state.get_agent_position(o) is None]
            if not cands:
                continue
            o = max(cands, key=lambda o: sum(p for c, p in self.belief[o].items() if manhattan(c, f) <= 1) + 1e-9
                    - 1e-6 * min(manhattan(c, f) for c in self.belief[o]))
            self.belief[o] = {f: 1.0}
            self.eaten_hint[o] = f

    def best_guess(self, o):
        b = self.belief[o]
        return max(b, key=b.get)

    def prob_near(self, o, cell, r):
        return sum(p for c, p in self.belief[o].items() if manhattan(c, cell) <= r)


# --------------------------------------------------------------------- A* search
def astar(m, start, goals, cost_fn=None, max_expand=4000):
    """A* from start to the nearest of `goals` (a set). cost_fn(cell)->extra cost >= 0.
    Heuristic: true maze distance to the closest goal (exact without extra costs,
    hence admissible because extra costs are non-negative). Returns list of cells
    (excluding start) or None."""
    if not goals:
        return None
    goals = set(goals)
    if start in goals:
        return []
    D, idx = m.D, m.idx
    gidx = [idx[g] for g in goals]

    def h(c):
        row = D[idx[c]]
        return min(row[g] for g in gidx)

    openh = [(h(start), 0.0, start)]
    g = {start: 0.0}
    parent = {start: None}
    expanded = 0
    while openh:
        f, gc, c = heapq.heappop(openh)
        if gc > g.get(c, 1e18):
            continue
        if c in goals:
            path = []
            while c != start:
                path.append(c)
                c = parent[c]
            return path[::-1]
        expanded += 1
        if expanded > max_expand:
            break
        for n in m.nbrs[c]:
            ng = gc + 1.0 + (cost_fn(n) if cost_fn else 0.0)
            if ng < g.get(n, 1e18):
                g[n] = ng
                parent[n] = c
                heapq.heappush(openh, (ng + h(n), ng, n))
    return None


# ----------------------------------------------------------------- forward model
class SimState:
    """Minimal forward model.  `pos` maps agent index -> cell for every agent we model
    (unknown opponents can simply be left out).  Food sets are split by the side they
    lie on, so either team can eat."""
    __slots__ = ('pos', 'carry', 'scared', 'food_red', 'food_blue', 'caps_red', 'caps_blue',
                 'score', 'deaths', 'kills', 'eaten', 'returned')

    def copy(self):
        s = SimState.__new__(SimState)
        s.pos = dict(self.pos)
        s.carry = dict(self.carry)
        s.scared = dict(self.scared)
        s.food_red, s.food_blue = self.food_red, self.food_blue
        s.caps_red, s.caps_blue = self.caps_red, self.caps_blue
        s.score, s.deaths, s.kills = self.score, dict(self.deaths), dict(self.kills)
        s.eaten, s.returned = dict(self.eaten), dict(self.returned)
        return s


class Sim:
    """Rules for SimState, from the point of view of team `red`.  score is red-positive."""

    def __init__(self, m, red_indices, starts):
        self.m = m
        self.red_idx = set(red_indices)
        self.starts = starts

    def is_red(self, a):
        return a in self.red_idx

    @staticmethod
    def from_game(game_state, m, tracker=None, include_unseen=True, max_unseen_dist=None, ref=None):
        sim = Sim(m, game_state.get_red_team_indices(),
                  {a: ipos(game_state.get_initial_agent_position(a)) for a in range(game_state.get_num_agents())})
        s = SimState.__new__(SimState)
        s.pos, s.carry, s.scared = {}, {}, {}
        for a in range(game_state.get_num_agents()):
            st = game_state.get_agent_state(a)
            p = ipos(st.get_position())
            if p is None and tracker is not None and include_unseen and a in tracker.belief:
                p = tracker.best_guess(a)
                if max_unseen_dist is not None and ref is not None and m.dist(p, ref) > max_unseen_dist:
                    p = None
            if p is not None:
                s.pos[a] = p
                s.carry[a] = st.num_carrying
                s.scared[a] = st.scared_timer
        s.food_red = frozenset(game_state.get_red_food().as_list())
        s.food_blue = frozenset(game_state.get_blue_food().as_list())
        s.caps_red = frozenset(game_state.get_red_capsules())
        s.caps_blue = frozenset(game_state.get_blue_capsules())
        s.score = game_state.get_score()
        s.deaths = {a: 0 for a in s.pos}
        s.kills = {a: 0 for a in s.pos}
        s.eaten = {a: 0 for a in s.pos}
        s.returned = {a: 0 for a in s.pos}
        return sim, s

    def legal(self, s, a):
        p = s.pos[a]
        return [action_to(p, n) for n in self.m.nbrs[p]] + ['Stop']

    def succ_cells(self, s, a):
        p = s.pos[a]
        return self.m.nbrs[p] + [p]

    def is_pacman(self, a, cell):
        return self.m.is_red_side(cell) != self.is_red(a)

    def move(self, s, a, cell):
        """Return a new state where agent a moved to (adjacent) cell."""
        s = s.copy()
        red = self.is_red(a)
        s.pos[a] = cell
        m = self.m
        pac = self.is_pacman(a, cell)
        if pac:
            if red and cell in s.food_blue:
                s.food_blue = s.food_blue - {cell}
                s.carry[a] += 1
                s.eaten[a] += 1
            elif (not red) and cell in s.food_red:
                s.food_red = s.food_red - {cell}
                s.carry[a] += 1
                s.eaten[a] += 1
            caps = s.caps_blue if red else s.caps_red
            if cell in caps:
                if red:
                    s.caps_blue = caps - {cell}
                else:
                    s.caps_red = caps - {cell}
                for b in s.pos:
                    if self.is_red(b) != red:
                        s.scared[b] = SCARED_TIME
        elif s.carry[a] > 0:
            s.score += s.carry[a] if red else -s.carry[a]
            s.returned[a] += s.carry[a]
            s.carry[a] = 0
        # collisions (same cell only, as in capture.py)
        for b, pb in list(s.pos.items()):
            if pb != cell or self.is_red(b) == red:
                continue
            b_pac = self.is_pacman(b, pb)
            if pac and not b_pac:          # I am pacman, b is ghost
                if s.scared.get(b, 0) > 0:
                    self._kill(s, b)
                    s.kills[a] += 1
                else:
                    self._kill(s, a)
                    s.kills[b] += 1
                    break
            elif (not pac) and b_pac:      # I am ghost, b is pacman
                if s.scared.get(a, 0) > 0:
                    self._kill(s, a)
                    s.kills[b] += 1
                    break
                else:
                    self._kill(s, b)
                    s.kills[a] += 1
        if s.scared.get(a, 0) > 0:
            s.scared[a] -= 1
        return s

    def _kill(self, s, a):
        # carried food is dropped back (approximation: returned to the board at the death cell's side)
        s.deaths[a] += 1
        s.carry[a] = 0
        s.pos[a] = self.starts[a]
        s.scared[a] = 0


# ------------------------------------------------------------------ agent base
import os
import sys
import traceback

from contest.capture_agents import CaptureAgent


class BaseAgent(CaptureAgent):
    """Common plumbing: map analysis, shared tracker, timing statistics, crash safety.
    Subclasses implement act(game_state) -> action."""
    TIME_BUDGET = 0.75          # seconds we allow ourselves per move (limit is 1.0), from the
                                # start of choose_action (time.perf_counter, monotonic)

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.move_times = []
        self.errors = 0

    def register_initial_state(self, game_state):
        t0 = time.perf_counter()
        # We do not need the framework's distancer (we compute our own table),
        # so we only replicate the cheap part of CaptureAgent.register_initial_state.
        self.red = game_state.is_on_red_team(self.index)
        self.register_team(self.get_team(game_state))
        self.m = MapInfo.get(game_state)
        self.start = ipos(game_state.get_agent_position(self.index))
        self.tracker = Tracker.get(self, game_state, fresh=self.index == min(self.get_team(game_state)))
        self.move_times = []
        self.startup_time = time.perf_counter() - t0
        self.setup(game_state)

    def get_maze_distance(self, a, b):
        return self.m.dist(ipos(a), ipos(b))

    def setup(self, game_state):
        pass

    def choose_action(self, game_state):
        t0 = time.perf_counter()
        self.t_start = t0
        legal = game_state.get_legal_actions(self.index)
        try:
            self.tracker = Tracker.get(self, game_state)
            self.tracker.update(self, game_state)
            a = self.act(game_state)
        except Exception:
            self.errors += 1
            if self.errors <= 3:
                traceback.print_exc(file=sys.stderr)
                log = os.environ.get('UPF_FC_ERRLOG')
                if log:
                    with open(log, 'a') as fh:
                        fh.write(f'--- {type(self).__name__} idx={self.index}\n' + traceback.format_exc())
            a = None
        if a not in legal:
            try:
                a = self.fallback_action(game_state, legal)
            except Exception:
                a = None
        if a not in legal:
            a = random.choice(legal)
        self.move_times.append(time.perf_counter() - t0)
        return a

    def fallback_action(self, game_state, legal):
        """Used when act() fails. The submitted my_team.py overrides this with the staff
        template's reflex policy (ReflexCaptureAgent.choose_action)."""
        return random.choice(legal)

    def time_left(self):
        return self.TIME_BUDGET - (time.perf_counter() - self.t_start)

    # ---- shared helpers
    def my_pos(self, gs):
        return ipos(gs.get_agent_position(self.index))

    def home_dist(self, p):
        return min(self.m.dist(p, b) for b in self.m.home_border(self.red))

    def visible_ghosts(self, gs):
        """Opponents that are ghosts (on their side) and visible: list of (pos, scared_timer)."""
        out = []
        for o in self.get_opponents(gs):
            st = gs.get_agent_state(o)
            p = ipos(st.get_position())
            if p is not None and not st.is_pacman:
                out.append((p, st.scared_timer, o))
        return out

    def invaders(self, gs):
        """Opponents currently on our side: list of (o, pos_or_guess, visible)."""
        out = []
        for o in self.get_opponents(gs):
            st = gs.get_agent_state(o)
            if st.is_pacman:
                p = ipos(st.get_position())
                out.append((o, p if p is not None else self.tracker.best_guess(o), p is not None))
        return out

    def moves_left(self, gs):
        return gs.data.timeleft // 4

# ========================= our code: agents/a2_astar.py =========================

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

# ========================= our code: agents/a6_iw.py =========================

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




class IWOffense(PlannerOffense):
    TIME_BUDGET = 0.30   # the 6,000-node cap normally stops the search at ~0.1 s; this only binds on a busy machine
    GAMMA = 0.98
    MAX_NODES = 6000
    USE_CARRY_ATOM = True
    LEAF_W = 0.3       # leaf penalty per carried pellet per square from home
    STUCK_TURNS = 20   # stalemate breaker (0 = off): see unstick()
    STUCK_PUSH = 8     # moves of plain shortest-path play once a stalemate is detected

    def unstick(self, gs):
        """Stalemate breaker. A ghost parked next to our Pacman can block it for hundreds
        of moves: every way out looks like death to the search, so it waits, and waiting
        earns nothing. After STUCK_TURNS of our moves on at most 2 cells, as a Pacman with
        no change in food carried, we follow the plain shortest path (home if carrying,
        else to food) for STUCK_PUSH moves. Returns an action, or None."""
        if not self.STUCK_TURNS:
            return None
        st = gs.get_agent_state(self.index)
        me = self.my_pos(gs)
        h = getattr(self, '_hist', [])
        h.append((me, st.num_carrying, st.is_pacman))
        self._hist = h = h[-self.STUCK_TURNS:]
        push = getattr(self, '_push', 0)
        if push == 0 and len(h) == self.STUCK_TURNS and all(x[2] for x in h) \
                and len({x[0] for x in h}) <= 2 and len({x[1] for x in h}) == 1:
            push = self.STUCK_PUSH
            self._hist = []
        if push == 0 or not st.is_pacman:
            self._push = 0
            return None
        self._push = push - 1
        m = self.m
        goals = set(m.home_border(self.red)) if st.num_carrying > 0 else set(self.get_food(gs).as_list())
        path = astar(m, me, goals)
        return action_to(me, path[0]) if path else None

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
        a = self.unstick(gs)
        if a is not None:
            return a
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


#####################
# Our team's agents #
#####################

class OffensiveIWAgent(IWOffense, OffensiveReflexAgent):
    """
    Offensive agent (approach A6 in REPORT.md): width-based lookahead, IW(1) with
    novelty atoms (cell, carried food) and (capsules left), on a fast forward model of
    the game in which the opponents' ghosts chase greedily. Each move we play the first
    action towards the node with the best discounted return; when no positive return is
    in reach we use goal selection + A* (approach A2). HMM tracking of the opponents.
    If anything fails, the staff template's OffensiveReflexAgent policy is used.
    """

    def choose_action(self, game_state):
        """Returns the next action (timing and crash safety in BaseAgent.choose_action)."""
        return BaseAgent.choose_action(self, game_state)

    def fallback_action(self, game_state, legal):
        return OffensiveReflexAgent.choose_action(self, game_state)


class DefensiveInterceptAgent(PlannerDefense, DefensiveReflexAgent):
    """
    Defensive agent (approach A2): predicts the invader's greedy path to our food (or
    back home) from the HMM belief and intercepts it at the first predicted cell it can
    reach in time; chases visible invaders and patrols the border otherwise.
    If anything fails, the staff template's DefensiveReflexAgent policy is used.
    """

    def choose_action(self, game_state):
        """Returns the next action (timing and crash safety in BaseAgent.choose_action)."""
        return BaseAgent.choose_action(self, game_state)

    def fallback_action(self, game_state, legal):
        return DefensiveReflexAgent.choose_action(self, game_state)
