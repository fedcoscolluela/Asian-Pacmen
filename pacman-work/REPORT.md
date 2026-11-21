# Pacman Capture-the-Flag: approaches, experiments and lessons

**Team Asian-Pacmen** (Jan Federico Coscolluela IV, Ittehad Chowdhury). Autonomous Systems, UPF, EMAI 2026-27. Setup and run commands are in [`README.md`](README.md).

**Submission:** `../my_team.py`. It combines **A6**, a width-based IW(1) attacker, with the **A2** A\*/HMM interception defender, on top of the staff template. Results:
* First in a 9-team round-robin of our approaches, with 81% of points.
* 392-46-90 against a 12-opponent pool and 628-84-36 against 17 teams from last edition.
* Average move 13 ms; worst 356 ms with the machine fully loaded (limit 1 s).
* No errors, time warnings or forfeits in any final test.

---

## 1. Shared infrastructure (`agents/upf_fc_core.py`)

All approaches share one toolkit, so they differ only in how they decide.

* **Map analysis:** all-pairs maze distances by BFS, home-border cells, distance home, and a dead-end pass giving each cell its depth (for trap avoidance).
* **Opponent tracking (HMM forward filter):** a belief over cells for each opponent, propagated with uniform moves, weighted by the noisy distance reading, cleared where we would see it, and restricted to the correct side by `is_pacman`. It snaps to a cell where our food vanished and resets to spawn on death. Both teammates share one tracker.
* **Forward model `Sim`:** a light copy of the rules (moves, eating, capsules, scared timers, deposits, collisions, respawn), much cheaper than `generate_successor`. This is what makes search possible within 1 s.
* **Safety (`BaseAgent`):** every move runs in try/except with a time budget. On error the agent falls back to the staff template's reflex policy, then to a random legal move.

## 2. Approaches

| id | file | idea |
|---|---|---|
| A1 | `a1_reflex.py` | One-step lookahead with hand-tuned features (simple baseline) |
| A2 | `a2_astar.py` | Goal selection (food, home, capsule, flee) + **A\*** with a danger-aware cost and maze-distance heuristic. **Defender:** predicts the invader's path and intercepts it at the first reachable cell; otherwise patrols near the HMM estimate |
| A3 | `a3_alphabeta.py`, `a3_expectimax.py`, `a3_ab_offense_only.py` | Near opponents: **alpha-beta** or **expectimax** on `Sim` with iterative deepening; otherwise A2 |
| A4 | `a4_mcts.py` | Same trigger, **MCTS/UCT** (C=60, 3-round rollouts) |
| A5 | `a5_qlearn.py` + variants | **Approximate Q-learning** (12 linear features), trained by self-play on 66 layouts |
| A6 | `a6_iw.py` | **IW(1)** on `Sim`: breadth-first, pruning nodes whose atom (cell, carried food) or (capsules left) is not new. Opponents chase greedily; nodes score a discounted return (+10 food, +100 deposit, +40 capsule, −600 death). Plays the first step to the best node (IW for simulators, Lipovetzky et al. 2015); falls back to A2 |
| A7 | `a7_width.py` + `a7_*.py` | The Lecture 8 family over a factored state (x, y, carry, capsules, ghost distance): IW(1), IW(2), Iterated IW, BFWS, and "pure" versions without the A2 fallback |
| A8 | `a8_hidden.py` | A6 + A2 reading the opponents' public hidden-state fields (§6) |
| A9 | `a9_win.py` | Fixes from our loss analysis (§7) |
| A10 | `a10_raid.py` | Defender that raids when nobody invades (§7) |

## 3. Results

`experiments/run_experiments.py` plays headless games under tournament rules (a move over 1 s is a warning; the third warning or any move over 3 s forfeits), each pairing on every layout with both colours. All CSVs are in `results/`.

**Against the staff baseline** (20 layouts × 2 colours = 40 games each):

| approach | W-T-L | mean score | max ms |
|---|---|---|---|
| A1 reflex | 35-4-1 | +7.8 | 132 |
| A2 A\* | 36-3-1 | +20.4 | 160 |
| A3 alpha-beta | 34-5-1 | +15.7 | 635 |
| A3 expectimax | 28-11-1 | +12.5 | 501 |
| A4 MCTS | 33-5-2 | +18.3 | 518 |
| A5 Q-learning from zero | 0-39-1 | 0.0 | 8 |
| A5c Q-learning from a prior | 37-1-2 | +21.1 | 9 |
| **A6 IW(1), (cell, carry) atom** | **38-1-1** | **+23.2** | 271 |
| A6 ablation: cell-only atom | 38-1-1 | +13.6 | 115 |
| A7 IW(2) | 38-1-1 | +17.0 | 95 |
| A7 BFWS | 37-3-0 | +12.3 | 174 |
| A7 pure IW(1), no fallback | 2-38-0 | +0.1 | 70 |
| A7 pure IW(2), no fallback | 38-1-1 | +16.4 | 290 |

The baseline is too weak to separate the good agents, so the round-robin is the real ranking.

**Round-robin** (9 teams, 10 layouts × 2 colours per pair, 720 games; win 3, tie 1):

| rank | team | W-T-L | points |
|---|---|---|---|
| 1 | **A6 IW(1)** | 125-15-20 | **81%** |
| 2 | A7 IW(2) | 96-19-45 | 64% |
| 3 | A7 Iterated IW | 86-27-47 | 59% |
| 4 | A7 BFWS | 83-26-51 | 57% |
| 5 | A2 A\* | 53-45-62 | 42% |
| 6 | A5c Q-learning | 57-33-70 | 42% |
| 7 | A3 alpha-beta | 22-51-87 | 24% |
| 8 | A4 MCTS | 24-41-95 | 24% |
| 9 | A1 reflex | 9-73-78 | 21% |

A6 beats every other team head-to-head (70–90% wins). Full table: `results/round_robin_summary.md`.

## 4. What failed, and why

1. **A1 reflex ties.** One step of lookahead cannot plan "eat and come back": 73 of 160 round-robin games were ties.
2. **Adversarial search (A3, A4) lost to plain A2** (15% and 5% wins against it). In ~0.45 s of Python they reach only 3–6 rounds, too shallow to see a deposit, so the cruder leaf evaluation dominates. Minimax opponents make the agent too cautious; expectimax makes it reckless. They were also 100–200× slower and the only approaches near the time limit (895 ms).
3. **Q-learning from zero learned nothing** in 200 episodes: scoring needs 10–30 correct steps and a deposit, which ε-exploration almost never finds.
4. **Reward-shaping bug.** With Φ = −distance and γ < 1, standing still earned a positive shaping reward, so the agent learned to idle. A potential Φ ≥ 0 fixed it, as Ng et al. (1999) require.
5. **Divergence (deadly triad).** Learning from a hand-set prior improved over 90 episodes, then collapsed to 0 wins by episode 150 as the weights blew up (function approximation + bootstrapping + off-policy). We kept early stopping at 90; the untrained prior was about as good, so most of the strength came from the features.
6. **Pure IW(1) cannot score** (2-38-0). "Eat, then return home" is a **width-2** problem, like Lecture 8's key-and-corridor example: the way home revisits cells already seen, so IW(1) prunes it. IW(2) solves it (38-1-1), and so does A6's joint (cell, carry) atom inside IW(1).
7. **Why A6 beats IW(2).** With 5 factors, IW(2) tracks 10 atom pairs, so under the same budget it searches wider but shallower. A6's single joint atom keeps the search deep along what matters. Choosing the right atoms mattered more than the width bound.
8. **Safety nets hide weaknesses.** With the A2 fallback on, weak variants look fine on wins and only differ in mean score. Ablations must be run without the fallback.
9. **Bugs found by experiments:** a tracker cache keyed by `id(layout)` leaked between games, and a chase-only defender never caught fleeing invaders. Path-prediction interception took RANDOM2 from −14 to +28.

## 5. Tuning and robustness

* **Node cap** (2k / 6k / 15k / 40k nodes): 48-8-16, 51-9-12, 51-9-12, 51-9-12 over 72 games. With 15k or 40k, 64 of 72 games are move-for-move identical, because IW(1) can only expand nodes that make a new atom true and usually runs out before the cap: the polynomial bound on IW(k) from Lecture 8. 2k cuts the horizon and loses games, so **6k is kept**.
* **Overloaded machine** (12 parallel games on 8 cores, 168 games incl. 30 unseen random layouts): 0 errors, warnings or forfeits; startup at most 0.21 s of the 15 s allowance.
* **Self-play** (37-10-37) is symmetric: no colour-dependent bug, and two copies of our team do not interfere.
* **Forced failure** (every planner call raises): the game finishes on the template reflex fallback, with no forfeit.

## 6. A8: reading the opponents' hidden state

An opponent more than 5 squares away has only its position hidden; `get_agent_state(i)` still gives its `is_pacman`, `num_carrying`, `num_returned` and `scared_timer`. The submitted agent already used `is_pacman` (tracking, invader detection), `num_carrying` (predicting a run home) and `scared_timer` (threat model). A8 added:
* **Tracker events:** changes in these fields pin a hidden opponent to our border (it just crossed), its own border (it deposited or left), or its start cell (it was eaten).
* **Defender:** goes for the invader carrying the most food.
* **Attacker:** keeps eating while every ghost stays scared longer than our trip home.

| test (A8 vs submitted) | result |
|---|---|
| vs baseline, A2, A7-IW(2), A5c (144 games) | identical, move for move |
| head to head (72 games) | 31-10-31 |
| vs two-attacker teams (72 games each) | 52-3-17 vs 51-3-18 |

The events were always correct but rarely mattered: most crossings happen within sight anyway. **Not submitted.** Build with `build_pacman_agent.py OUT.py --hidden`.

## 7. A9: loss analysis and final tests

**Test pool:** the same games for every version, on 22 layouts × 2 colours, against:
* **our 12 opponents:** A2, A3, A4, A5c, A7-IW(2), A8, the baseline and five stress teams in `agents/test_*.py` (two attackers, two turtles);
* **17 teams from last edition,** taken from the repositories shared on the course forum (2 of 19 were offline). We only changed their imports and checked their code first. Their code is not in this repo.

**Losses of the submitted agent:**
* after a capsule, the scared defender only keeps its distance;
* one defender cannot stop two attackers;
* many ties on tiny maps and against turtles.

A9 (`a9_win.py`, built with `--win`) adds:
* a scared defender waits on the invader's way home;
* the defender guards a capsule the invader is about to reach;
* the attacker comes home when both opponents invade, or in the last 40 moves with a lead.

| version | own pool (528 games) | past teams (748 games) | points per game vs submitted |
|---|---|---|---|
| **submitted** | **392-46-90** | **628-84-36** | — |
| A9 defender fixes only | 386-43-99 | 627-81-40 | −0.04 ± 0.04 (noise) |
| A9 full | 360-48-120 | 616-83-49 | **−0.18 ± 0.05** |

Full A9 collapses against double attackers (8-4-32 vs 29-3-12): with our attacker home, nobody scores, and both invaders still get through. **Not submitted.**

**Time budget** (in the submitted agent): all timing uses `time.perf_counter()` from the start of each move; 0.75 s per move overall; the IW search stops at 0.30 s (checked every 32 nodes) and returns its best action so far; path prediction stops when less than 0.4 s is left. It changed no results (392 wins with and without it) and cut the worst move under load from 869 ms to 356 ms.

**A10: a defender that raids** (`a10_raid.py`, built with `--raid`). This targets the "defender never raids" limitation. While no opponent is in our half, the defender makes short raids with the attacker's IW(1) search. It turns back after 2 pellets or 5 steps from the border, and runs home as soon as an opponent crosses. In the last 60 moves of a game we are not winning, the limits loosen to 5 pellets and 12 steps.

| version | own pool (528 games) | points per game vs submitted |
|---|---|---|
| **submitted** | **392-46-90** | — |
| A10 | 330-61-137 | **−0.32 ± 0.06** (better in 54 games, worse in 111) |

A10 beat the turtle team more often (37-7-0 vs 34-10-0) but lost elsewhere, worst against double attackers (17-6-21 vs 29-3-12) and the Q-learning team (25-2-17 vs 36-4-4). The defender only turns back once an opponent has crossed, by which time it is several steps away and the invader gets in free. Raiding also runs the search on both agents: A10 had 2 moves over 1 s (warnings) and a worst move of 1.17 s. **Not submitted.** A safer version would raid only after detecting that the opponents never invade.

| move times, full load (7 games on 8 cores) | moves | mean | p99 | max | over 1 s |
|---|---|---|---|---|---|
| submitted, own pool | 304,425 | 13 ms | 87 ms | 356 ms | 0 |
| submitted, past teams | 426,306 | 14 ms | 93 ms | 267 ms | 0 |

## 8. Limitations

* Only the attacker uses search; the defender is the same A2 interceptor in every approach.
* Roles are fixed: no teammate coordination, and the defender never raids (A10 tried raiding and lost more than it gained).
* Opponent ghosts are modelled as greedy chasers only.
* Timing was measured on a Mac, not on the course cluster.
