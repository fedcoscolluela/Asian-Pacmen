"""
Play one headless game and print an event log (deaths, deposits, capsules) -- used to
diagnose why an approach loses on a layout.

usage (from pacman-work/):  python experiments/trace_game.py a2_astar baseline RANDOM2 [seed]
"""
import io
import os
import random
import sys
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiments import team_path, layout_arg  # noqa: E402


def main():
    red, blue, lay = sys.argv[1:4]
    seed = sys.argv[4] if len(sys.argv) > 4 else '1000'
    import contest.capture as capture
    random.seed(seed)
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        opts = capture.read_command(['-r', team_path(red), '-b', team_path(blue), '-l', layout_arg(lay), '-Q',
                                     '--delay-step', '0', '--setRandomSeed', seed])
        g = capture.run_games(**opts)[0]
    # replay
    from contest.capture import GameState
    st = GameState()
    st.initialize(opts["layouts"][0], 4)
    st.data.timeleft = 1200
    names = {0: 'R0', 1: 'B1', 2: 'R2', 3: 'B3'}
    t = 0
    for (i, a) in g.move_history:
        prev = st
        st = st.generate_successor(i, a)
        t += 1
        for j in range(4):
            p0, p1 = prev.get_agent_position(j), st.get_agent_position(j)
            s0, s1 = prev.get_agent_state(j), st.get_agent_state(j)
            if p0 and p1 and abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) > 1:
                print(f't={t:4d} {names[j]} DIED at {p0} carrying {s0.num_carrying}')
            if s1.num_returned > s0.num_returned:
                print(f't={t:4d} {names[j]} deposited {s1.num_returned - s0.num_returned}  score={st.get_score()}')
        if len(st.get_red_capsules()) + len(st.get_blue_capsules()) < len(prev.get_red_capsules()) + len(prev.get_blue_capsules()):
            print(f't={t:4d} {names[i]} ate a capsule')
    print('final score', g.state.data.score, 'moves', len(g.move_history))


if __name__ == '__main__':
    main()
