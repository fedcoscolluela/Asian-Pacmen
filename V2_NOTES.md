# v2 (k5): what this branch changes

`my_team.py` = Jan's `my_team.py` at commit 21c320e, unchanged, with an override block appended
at the end (`V2Attacker`, `V2Defender`, and `create_team` now uses them). The diff is additions only.
A `FLAGS` dict at the bottom switches each change on or off; all off = Jan's agent at 21c320e.

## What is switched on

1. `duel_def`: at startup the defender solves one ghost vs one invader on our half exactly
   (retrograde analysis over ghost cell, invader cell, and side to move). If a visible invader
   can no longer get home, the defender plays the move that keeps it trapped.
2. `duel_delay`: if a carrying invader *can* escape, the defender picks the move that makes
   the escape take longest.
3. `bank_delay` (new in k5): for an *empty* invader, the defender solves how many moves the
   invader needs to eat a pellet and get it home, and plays to make that as long as possible.
   This is solved exactly with a min-max shortest-path search, takes ~0.04 s, and is rebuilt
   whenever our food changes.
4. `duel_rebuild`: the tables are rebuilt after one of our capsules is eaten.
5. `late_allin`: when we are behind in the last 200 moves and nobody is invading, the defender
   attacks too, using the IW planner.

## Test results (local, layouts not used for tuning)

* Head-to-head vs 21c320e: 46-9-25 over 80 games, 1.84 points/game (95% CI 1.55-2.11;
  1.50 = even).
* vs the 12 hardest of 43 public framework teams: +0.24 points/game over g3, the previous
  version without `bank_delay`/`duel_rebuild`, which itself was +0.44 over 21c320e.
* All 43 teams: about 87% of points. 0 crashes, 0 timing warnings, slowest move 0.27 s.

Not yet tested against v12 (trapping defender, 6e9e142), which was pushed after this work started.

Run it like any team: `python -m contest.capture -r my_team.py -b <other>.py`
