"""
Build the submission my_team.py ON TOP OF the staff template (aig-upf/pacman-agent/my_team.py).

The output keeps the template's header, imports, create_team (docstring included) and its
three agent classes (ReflexCaptureAgent, OffensiveReflexAgent, DefensiveReflexAgent)
unchanged; only create_team's default agent names are switched to ours. The template
reflex agents are then used as our agents' FALLBACK policy (if the planner ever throws).
Our planner code (upf_fc_core + a2_astar + a6_iw) is appended, followed by the two team
classes, which inherit from CaptureAgent and override choose_action as the README asks.

usage (from pacman-work/):
  python experiments/build_pacman_agent.py ../my_team.py
  python experiments/build_pacman_agent.py OUT.py --hidden   # A8: also reads the opponents' hidden state
  python experiments/build_pacman_agent.py OUT.py --win      # A9: fixes from the contest-readiness review
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_submission import AG, ROOT, strip  # noqa: E402

TEMPLATE = os.path.join(ROOT, 'template', 'my_team_staff_template.py')
HIDDEN = '--hidden' in sys.argv
WIN = '--win' in sys.argv
RAID = '--raid' in sys.argv
PARTS = ['upf_fc_core', 'a2_astar', 'a6_iw'] + (['a8_hidden'] if HIDDEN else []) + (['a9_win'] if WIN else []) + (['a10_raid'] if RAID else [])
OFF, DEF = 'OffensiveIWAgent', 'DefensiveInterceptAgent'
OFF_BASE = 'WinOffense' if WIN else 'IWOffenseHidden' if HIDDEN else 'IWOffense'
DEF_BASE = 'RaidDefense' if RAID else 'WinDefense' if WIN else 'PlannerDefenseHidden' if HIDDEN else 'PlannerDefense'
HIDDEN_NOTE = ('''
    Approach A10: when nobody invades, the defender makes short leashed raids.''' if RAID else ('''
    Approach A9 (REPORT.md section 10): scared defender guards the invader's way home,
    capsule guarding, attacker helps defend against two invaders and protects a late lead.'''
               if WIN else '''
    Approach A8: also reads the opponents' public is_pacman / num_carrying / num_returned /
    scared_timer fields, which stay visible when their position is hidden.''' if HIDDEN else ''))

TEAM_CLASSES = f'''

#####################
# Our team's agents #
#####################

class {OFF}({OFF_BASE}, OffensiveReflexAgent):
    """
    Offensive agent (approach A6 in REPORT.md): width-based lookahead, IW(1) with
    novelty atoms (cell, carried food) and (capsules left), on a fast forward model of
    the game in which the opponents' ghosts chase greedily. Each move we play the first
    action towards the node with the best discounted return; when no positive return is
    in reach we use goal selection + A* (approach A2). HMM tracking of the opponents.
    If anything fails, the staff template's OffensiveReflexAgent policy is used.{HIDDEN_NOTE}
    """

    def choose_action(self, game_state):
        """Returns the next action (timing and crash safety in BaseAgent.choose_action)."""
        return BaseAgent.choose_action(self, game_state)

    def fallback_action(self, game_state, legal):
        return OffensiveReflexAgent.choose_action(self, game_state)


class {DEF}({DEF_BASE}, DefensiveReflexAgent):
    """
    Defensive agent (approach A2): predicts the invader's greedy path to our food (or
    back home) from the HMM belief and intercepts it at the first predicted cell it can
    reach in time; chases visible invaders and patrols the border otherwise.
    If anything fails, the staff template's DefensiveReflexAgent policy is used.{HIDDEN_NOTE}
    """

    def choose_action(self, game_state):
        """Returns the next action (timing and crash safety in BaseAgent.choose_action)."""
        return BaseAgent.choose_action(self, game_state)

    def fallback_action(self, game_state, legal):
        return DefensiveReflexAgent.choose_action(self, game_state)
'''


def main():
    dest = [a for a in sys.argv[1:] if not a.startswith('--')][0]
    tpl = open(TEMPLATE).read().replace('\r', '')
    head, rest = tpl.split('#################\n# Team creation #', 1)
    team_part, agents_part = rest.split('##########\n# Agents #', 1)
    team_part = '#################\n# Team creation #' + team_part
    n = 0
    team_part, n = re.subn(r"first='OffensiveReflexAgent', second='DefensiveReflexAgent'",
                           f"first='{OFF}', second='{DEF}'", team_part)
    assert n == 1, 'template create_team signature not found'
    agents_part = '##########\n# Agents #' + agents_part

    out = [head.rstrip() + '\n',
           '# --- added by our team (the planner code below needs these) ---',
           'import os', 'import sys', 'import time', '',
           '# my_team.py for the UPF Pacman contest, built by pacman-work/experiments/build_pacman_agent.py',
           '# from the staff template + pacman-work/agents/{' + ', '.join(PARTS) + '}.py.',
           '# Structure: [template header] [template create_team] [template reflex agents, unchanged,',
           '# used as fallback] [our toolkit and planners] [our team agents].',
           '#',
           '# ---------------- GUIDE FOR READERS: search for "[KEY-" to jump to each feature ----------------',
           '# Team: Asian-Pacmen. Attacker = IWOffense (width-based IW(1) search), defender = PlannerDefense (A* interception).',
           '#  [KEY-1]  all-pairs maze distances, precomputed at startup',
           '#  [KEY-2]  dead-end analysis of the map (trap avoidance)',
           '#  [KEY-3]  HMM belief tracking of unseen opponents',
           '#  [KEY-4]  A* search with a danger-aware cost',
           '#  [KEY-5]  own fast game simulator used by the search',
           '#  [KEY-6]  runtime safety: time budget + fallback, never forfeits',
           '#  [KEY-7]  risk-aware path cost near ghosts',
           '#  [KEY-8]  DEFENDER: predict invader path and intercept it',
           '#  [KEY-9]  ATTACKER: IW(1) width-based planning (main novelty, beyond lecture)',
           "#  [KEY-10] stalemate breaker (found by analysing losses to last edition's teams)",
           '#  [KEY-11] opponent ghost model inside the search',
           '# Full experiments, ablations and results: pacman-work/REPORT.md',
           '# -----------------------------------------------------------------------------------------------', '', '',
           team_part.rstrip() + '\n', '',
           '# The staff template agents below are kept unchanged: our agents fall back on them.',
           agents_part.rstrip() + '\n']
    for p in PARTS:
        src = open(os.path.join(AG, p + '.py')).read()
        out.append(f'\n# {"=" * 25} our code: agents/{p}.py {"=" * 25}\n')
        out.append(strip(src, drop_create_team=True))
    out.append(TEAM_CLASSES)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, 'w') as f:
        f.write('\n'.join(out).rstrip() + '\n')
    print('wrote', dest)


if __name__ == '__main__':
    main()
