# Pacman Capture-the-Flag — Autonomous Systems programming project (UPF, EMAI 2026-27)

Development folder of our submission repo for the project in the course PDF (`programming_project.pdf`). The write-up is in
[`REPORT.md`](REPORT.md); the **submission** is `../my_team.py` at the repo root: the official `aig-upf/pacman-agent` template repo, with the contest framework pulled as a submodule, and our agent built on top of the staff template.

## What is in here

| path | what |
|---|---|
| `../` (repo root) | **the submission repo**: the official `aig-upf/pacman-agent` template with the current `pacman-contest` framework as a git submodule (exactly as in PDF §3.1). Every script here uses that framework |
| `template/my_team_staff_template.py` | unmodified copy of the staff template `my_team.py`, which the submission is built on |
| `agents/upf_fc_core.py` | shared toolkit: map analysis (all-pairs maze distances, dead-end depth), HMM opponent tracker, A*, a fast forward model (`Sim`), and the agent base class (timing, crash safety) |
| `agents/a1_reflex.py` | **A1** improved reflex agents (1-step lookahead, hand-tuned features) — the simple baseline |
| `agents/a2_astar.py` | **A2** goal selection + A* with a danger-aware cost map; HMM-tracking interception defender |
| `agents/a3_alphabeta.py`, `a3_expectimax.py`, `a3_ab_offense_only.py` | **A3** depth-limited alpha-beta / expectimax near opponents (+ ablation) |
| `agents/a4_mcts.py` | **A4** Monte-Carlo tree search (UCT) near opponents |
| `agents/a5_qlearn.py`, `a5b_qlearn_shaped.py`, `a5c_qlearn_prior.py`, `a5p_linear_prior.py` (+ `*_weights.json`) | **A5** approximate Q-learning: from zero, with reward shaping, from a prior (+ untrained control) |
| `agents/a6_iw.py`, `a6_iw_pos.py` | **A6** width-based lookahead, IW(1) with a joint (cell, carried-food) atom (+ cell-only ablation). The node-cap tuning in REPORT §5 set `IWOffense.MAX_NODES` to 2k/15k/40k |
| `agents/a7_width.py` and `a7_*.py` | **A7** width-based family from Lecture 8 over a factored state: IW(1), IW(2), Iterated IW, BFWS, and "pure" ablations without fallback |
| `experiments/run_experiments.py` | parallel headless match runner → CSV + markdown summary (tournament rules: 1 s warnings, 3 s forfeit) |
| `experiments/train_qlearn.py` | trains the A5 agents (learning curves in `results/`) |
| `agents/a8_hidden.py` | **A8** reads the opponents' public hidden-state fields (is_pacman, num_carrying, scared_timer) — tested, not submitted (REPORT §6) |
| `agents/a9_win.py` | **A9** fixes from the loss analysis (scared guard, capsule guard, help-defend, protect-lead) — tested, not submitted (REPORT §7) |
| `agents/a10_raid.py` | **A10** defender that raids when nobody invades — tested, not submitted (REPORT §7) |
| `agents/test_*.py` | stress-test opponent teams (two attackers, two turtles, …) used in the REPORT §7 pool |
| `experiments/trace_game.py` | event log of one game (deaths, deposits, capsules), used for diagnosis |
| `experiments/build_pacman_agent.py` | builds the submission `my_team.py` (`--hidden` / `--win` / `--raid` build the A8 / A9 / A10 variants) on top of the staff template (`template/my_team_staff_template.py`, an unmodified copy of aig-upf/pacman-agent's `my_team.py`) |
| `experiments/build_submission.py` | bundling helpers used by `build_pacman_agent.py` (strips local imports; can also bundle any single approach into one file) |
| `results/` | every CSV / summary quoted in the report |

## Setup (macOS / Linux): the PDF's §3.1 steps, from the repo root

```bash
cd ..
```

```bash
python3 -m venv venv && source venv/bin/activate
```

```bash
cd pacman-contest && pip install -e . && pip install -r requirements.txt && cd ..
```

The PDF uses Python 3.8. We tested on 3.9 (with the pinned requirements) and 3.12; on 3.12 the
pinned `frozendict==2.3.4` may not build, in which case `pip install frozendict` works.

## Play / watch a game (from `../pacman-contest/src/contest`)

```bash
python capture.py -r baseline_team -b ../../../my_team.py
```

Any approach from this folder can be played too, e.g.
`-r ../../../pacman-work/agents/a7_iw2.py -b ../../../pacman-work/agents/a2_astar.py`,
on any layout (`-l officeCapture`, `-l RANDOM13`, …). Add `-q` for no graphics and `--delay-step 0` for speed.

## Reproduce the experiments (from this folder, with the venv active)

```bash
python experiments/run_experiments.py --teams a2_astar a6_iw --opponents baseline --layouts default fast office jumbo alley blox crowded distant medium strategic tiny RANDOM1 RANDOM2 RANDOM3 RANDOM4 RANDOM5 RANDOM6 RANDOM7 RANDOM8 RANDOM9 --workers 7 --out results/my_run.csv
```

```bash
python experiments/run_experiments.py --round-robin --teams a1_reflex a2_astar a6_iw a7_iw2 --layouts default office RANDOM1 --workers 7 --out results/my_rr.csv
```

```bash
python experiments/train_qlearn.py --episodes 90 --shaping --alpha 0.005 --eps 0.05 --out agents/a5c_weights.json --curve results/a5c_learning_curve.csv
```

(A5c starts from the weights already in `--out`; copy `results/a5c_prior_weights.json` there first to retrain from the prior.)

Rebuild the submission (staff template + our code) after changing `agents/`:

```bash
python experiments/build_pacman_agent.py ../my_team.py
```

To use another framework copy, set `UPF_FC_CONTEST_SRC=<path to pacman-contest/src>`. The results before the final re-tests (§3 of the report) were produced with last year's framework version (bf5a7d6). The current one (54cd83c) only adds speed-ups with identical game rules.

Note: `RANDOM<n>` layouts reseed Python's RNG when the maze is generated, so games on them are deterministic for a given pair of teams; the fixed layouts vary with `--seed`.
