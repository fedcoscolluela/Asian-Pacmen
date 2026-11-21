"""
Experiment harness: plays many headless games in parallel and writes a CSV + summary.

Every pairing is played with our team as RED and as BLUE (the map is symmetric but the
first mover and spawn side differ), on every requested layout, with fixed seeds so the
results are reproducible.  Games are run exactly like the tournament: catch_exceptions
is on, so >1 s moves give warnings, the 3rd warning or any 3 s move forfeits.

usage (from pacman-work/):
  python experiments/run_experiments.py --teams a1_reflex a2_astar --opponents baseline \
      --layouts default fast office RANDOM1 RANDOM2 --games 2 --workers 8 --out results/main.csv
Names: 'baseline' = staff baseline_team, otherwise agents/<name>.py
"""
import argparse
import csv
import json
import io
import multiprocessing as mp
import os
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the framework is the submodule of the submission repo (../pacman-agent), i.e. the current official version
# (works both next to the repo, ../pacman-agent, and inside it, pacman-agent/pacman-work)
_CANDIDATES = [os.path.join(os.path.dirname(ROOT), 'pacman-agent', 'pacman-contest', 'src'),
               os.path.join(os.path.dirname(ROOT), 'pacman-contest', 'src')]
CONTEST_SRC = os.environ.get('UPF_FC_CONTEST_SRC') or next(
    (c for c in _CANDIDATES if os.path.isdir(os.path.join(c, 'contest'))), _CANDIDATES[0])
LAYOUT_DIR = os.path.join(CONTEST_SRC, 'contest', 'layouts')
sys.path.insert(0, CONTEST_SRC)


def team_path(name):
    if name == 'baseline':
        return os.path.join(CONTEST_SRC, 'contest', 'baseline_team.py')
    if name.endswith('.py'):
        return os.path.abspath(name)
    return os.path.join(ROOT, 'agents', name + '.py')


def layout_arg(name):
    if name.upper().startswith('RANDOM'):
        return name.upper()
    if not name.endswith('Capture'):
        name = name + 'Capture'
    return os.path.join(LAYOUT_DIR, name)


def play(job):
    red, blue, layout, seed = job
    import contest.capture as capture
    random.seed(seed)
    args = ['-r', team_path(red), '-b', team_path(blue), '-l', layout_arg(layout), '-Q',
            '--delay-step', '0', '-c', '--setRandomSeed', str(seed)]
    buf = io.StringIO()
    t0 = time.time()
    with redirect_stdout(buf), redirect_stderr(buf):
        opts = capture.read_command(args)
        games = capture.run_games(**opts)
    wall = time.time() - t0
    g = games[0]
    score = g.state.data.score
    row = dict(red=red, blue=blue, layout=layout, seed=seed, score=score,
               crashed=int(bool(getattr(g, 'agent_crashed', False))),
               timeout=int(bool(getattr(g, 'agent_timeout', False))),
               warnings_red=sum(g.total_agent_time_warnings[i] for i in (0, 2)),
               warnings_blue=sum(g.total_agent_time_warnings[i] for i in (1, 3)),
               moves=len(g.move_history), wall=round(wall, 2))
    for side, idxs in (('red', (0, 2)), ('blue', (1, 3))):
        times = []
        errs = 0
        for i in idxs:
            times += getattr(g.agents[i], 'move_times', [])
            errs += getattr(g.agents[i], 'errors', 0)
        row[f'mean_ms_{side}'] = round(1000 * sum(times) / len(times), 1) if times else ''
        row[f'max_ms_{side}'] = round(1000 * max(times), 1) if times else ''
        row[f'errors_{side}'] = errs
        row[f'over700_{side}'] = sum(1 for t in times if t > 0.7)
        row[f'over1000_{side}'] = sum(1 for t in times if t > 1.0)
        row[f'times_{side}'] = times
        row[f'startup_ms_{side}'] = round(1000 * max(getattr(g.agents[i], 'startup_time', 0) for i in idxs), 1)
    out = buf.getvalue()
    row['crash_msg'] = ''
    if row['crashed']:
        tail = [l for l in out.splitlines() if 'Error' in l or 'crash' in l or 'timed out' in l or 'warning' in l]
        row['crash_msg'] = ' | '.join(tail[-3:])[:300]
    return row


def summarise(rows, teams):
    """Per team/opponent/layout: our win/tie/loss and mean score from OUR perspective."""
    agg = {}
    for r in rows:
        for us, them, sign in ((r['red'], r['blue'], 1), (r['blue'], r['red'], -1)):
            if us not in teams:
                continue
            s = sign * r['score']
            side = 'red' if sign == 1 else 'blue'
            k = (us, them)
            a = agg.setdefault(k, dict(n=0, w=0, t=0, l=0, score=0, maxms=0, meanms=[], crash=0))
            a['n'] += 1
            a['w'] += s > 0
            a['t'] += s == 0
            a['l'] += s < 0
            a['score'] += s
            if r[f'max_ms_{side}'] != '':
                a['maxms'] = max(a['maxms'], r[f'max_ms_{side}'])
                a['meanms'].append(r[f'mean_ms_{side}'])
            a['crash'] += r['crashed'] and s < 0
    lines = ['| team | opponent | games | W | T | L | win% | mean score | mean ms/move | max ms/move | forfeits |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for (us, them), a in sorted(agg.items()):
        mm = sum(a['meanms']) / len(a['meanms']) if a['meanms'] else 0
        lines.append(f"| {us} | {them} | {a['n']} | {a['w']} | {a['t']} | {a['l']} | "
                     f"{100 * a['w'] / a['n']:.0f}% | {a['score'] / a['n']:+.1f} | {mm:.1f} | {a['maxms']:.0f} | {a['crash']} |")
    # overall leaderboard with the tournament's scoring (win 3, tie 1, loss 0)
    tot = {}
    for (us, them), a in agg.items():
        t = tot.setdefault(us, dict(n=0, w=0, t=0, l=0, score=0))
        for k in ('n', 'w', 't', 'l', 'score'):
            t[k] += a[k]
    lines += ['', '| rank | team | games | W | T | L | points | points % | mean score |',
              '|---|---|---|---|---|---|---|---|---|']
    ranked = sorted(tot.items(), key=lambda kv: -(3 * kv[1]['w'] + kv[1]['t']) / kv[1]['n'])
    for i, (us, t) in enumerate(ranked, 1):
        pts = 3 * t['w'] + t['t']
        lines.append(f"| {i} | {us} | {t['n']} | {t['w']} | {t['t']} | {t['l']} | {pts} | "
                     f"{100 * pts / (3 * t['n']):.0f}% | {t['score'] / t['n']:+.1f} |")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teams', nargs='+', required=True)
    ap.add_argument('--opponents', nargs='+', default=['baseline'])
    ap.add_argument('--layouts', nargs='+', default=['default'])
    ap.add_argument('--games', type=int, default=1, help='games per (pair, layout, colour)')
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument('--out', default='results/run.csv')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--round-robin', action='store_true',
                    help='ignore --opponents: every unordered pair of --teams plays (both colours)')
    a = ap.parse_args()

    jobs = []
    if a.round_robin:
        pairs = [(t, o) for i, t in enumerate(a.teams) for o in a.teams[i + 1:]]
    else:
        pairs = [(t, o) for t in a.teams for o in a.opponents if o != t]
    for t, o in pairs:
        if True:
            for lay in a.layouts:
                for g in range(a.games):
                    seed = a.seed * 1000 + g
                    jobs.append((t, o, lay, seed))
                    jobs.append((o, t, lay, seed))
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f'{len(jobs)} games on {a.workers} workers -> {out}', flush=True)
    t0 = time.time()
    rows = []
    with mp.Pool(a.workers, maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(play, jobs)):
            rows.append(r)
            if (i + 1) % max(1, len(jobs) // 10) == 0:
                print(f'  {i + 1}/{len(jobs)} done ({time.time() - t0:.0f}s)', flush=True)
    rows.sort(key=lambda r: (r['red'], r['blue'], r['layout'], r['seed']))
    times = {}
    for r in rows:
        for side in ('red', 'blue'):
            times.setdefault(r[side], []).extend(r.pop(f'times_{side}'))
    with open(out.replace('.csv', '_times.json'), 'w') as f:
        json.dump({k: [round(t, 4) for t in v] for k, v in times.items()}, f)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = summarise(rows, set(a.teams))
    with open(out.replace('.csv', '_summary.md'), 'w') as f:
        f.write(summ + '\n')
    print(summ)
    print(f'total {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
