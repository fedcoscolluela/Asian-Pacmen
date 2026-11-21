"""
Train the A5 approximate Q-learning attacker.

usage (from pacman-work/):
  python experiments/train_qlearn.py --episodes 200 --out agents/a5_weights.json [--from-zero]

Each episode is one full game on a random training layout (RANDOM101.. seeds and the
fixed layouts), against a random opponent from --opponents, as red or blue at random.
Weights are written after every episode; the learning curve goes to results/a5_learning_curve.csv.
The RANDOM1..RANDOM9 maps used in the evaluation are never used for training.
"""
import argparse
import csv
import io
import os
import random
import sys
from contextlib import redirect_stdout, redirect_stderr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_experiments import ROOT, team_path, layout_arg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--out', default=os.path.join(ROOT, 'agents', 'a5_weights.json'))
    ap.add_argument('--curve', default=os.path.join(ROOT, 'results', 'a5_learning_curve.csv'))
    ap.add_argument('--opponents', nargs='+', default=['baseline', 'a2_astar'])
    ap.add_argument('--from-zero', action='store_true')
    ap.add_argument('--alpha', default='0.02')
    ap.add_argument('--eps', default='0.1')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--shaping', action='store_true', help='potential-based reward shaping')
    a = ap.parse_args()
    rng = random.Random(a.seed)   # private RNG: capture.read_command reseeds the global one
    out = os.path.abspath(a.out)
    os.environ['UPF_FC_TRAIN'] = '1'
    os.environ['UPF_FC_QWEIGHTS'] = out
    os.environ['UPF_FC_ALPHA'] = a.alpha
    os.environ['UPF_FC_EPS'] = a.eps
    if a.shaping:
        os.environ['UPF_FC_SHAPING'] = '1'
    if a.from_zero:
        os.environ['UPF_FC_QINIT'] = 'zero'
    train_layouts = ['default', 'fast', 'office', 'alley', 'medium', 'crowded', 'strategic'] + \
                    [f'RANDOM{100 + i}' for i in range(1, 60)]
    import contest.capture as capture
    os.makedirs(os.path.dirname(a.curve), exist_ok=True)
    rows = []
    for ep in range(a.episodes):
        lay = rng.choice(train_layouts)
        opp = rng.choice(a.opponents)
        we_red = rng.random() < 0.5
        red, blue = ('a5_qlearn', opp) if we_red else (opp, 'a5_qlearn')
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            opts = capture.read_command(['-r', team_path(red), '-b', team_path(blue), '-l', layout_arg(lay),
                                         '-Q', '--delay-step', '0', '--setRandomSeed', str(ep)])
            g = capture.run_games(**opts)[0]
        os.environ.pop('UPF_FC_QINIT', None)
        idx = 0 if we_red else 1
        agent = g.agents[idx]
        type(agent).save(out)
        score = g.state.data.score * (1 if we_red else -1)
        ep_r = type(agent).episode_log[-1][0] if type(agent).episode_log else 0.0
        rows.append(dict(episode=ep, layout=lay, opponent=opp, red=int(we_red), score=score,
                         reward=round(ep_r, 2), **{f'w_{k}': round(v, 4) for k, v in type(agent).weights.items()}))
        print(f'ep {ep:4d} {lay:10s} vs {opp:9s} score {score:+4d}  R={ep_r:8.1f}', flush=True)
        with open(a.curve, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
