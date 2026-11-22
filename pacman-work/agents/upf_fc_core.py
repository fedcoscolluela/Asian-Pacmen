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

    # ======== [KEY-1] ========
    # Precomputed all-pairs maze distances (BFS once at startup, inside the 15 s budget) -> O(1) distance lookups every move
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

    # ======== [KEY-2] ========
    # Dead-end / corridor analysis of the map: cells where a Pacman can be trapped; food inside one is only taken if we can get out before a ghost seals it
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
# ======== [KEY-3] ========
# HMM belief tracking (lecture: probabilistic state estimation) of unseen opponents: elapse = motion model, observe = noisy distance reading, plus hard evidence (eaten food, kills, sightings)
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
# ======== [KEY-4] ========
# A* search (lecture: heuristic search) with maze-distance heuristic and a pluggable danger cost; used by attacker, defender and stalemate breaker
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


# ======== [KEY-5] ========
# Own lightweight forward model of the capture game (much faster than GameState.generate_successor), what makes width-based search affordable within 1 s
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

    # ======== [KEY-6] ========
    # Runtime safety: every move is timed, the search stops at TIME_BUDGET (time_left), and any exception falls back to the staff reflex agent, so we never forfeit
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
