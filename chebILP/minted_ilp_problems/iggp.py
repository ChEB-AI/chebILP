"""IGGP-style minted problems (Inductive General Game Playing, Cropper, Evans & Law 2020).

Each example is one game state; the BK gives the fluents true in it and the move taken, and the
target is a ``next`` relation: the value a fluent takes in the following state.
"""
import itertools
import random
import string

from chebILP.ilp_problem import ILPProblem
from chebILP.minted_ilp_problems.base_minter import ILPProblemMinter


class IGGPMinter(ILPProblemMinter):
    """Shared, leakage-free task description: it names no game-specific fluent, action or stride."""

    target: str

    def get_task_description(self):
        return f"""\
IGGP (Inductive General Game Playing) turns a game specification into an ILP task. Each
example is one game STATE, named s0, s1, ...: the background gives the fluents true in that
state and the move taken in it, via does(State, robot, Action). The learning target is a
"next" relation -- the value a fluent takes in the state that FOLLOWS this one. The usable
fluent, action and auxiliary relations are defined in the glossary; a sample of background
facts shows the constant vocabulary. The exact effect of each move is deliberately NOT
disclosed -- you must infer it from the vocabulary and the relations provided.

The hidden rule reconstructs the target's next value from the current state and the move
taken. It is not given here; it typically has one or more clauses per action kind. The
dependence of the next value on the current state can range, across actions, from:
  - a direct copy of a current value;
  - a single step along a provided relation (a successor/predecessor, or a simple toggle);
  - a LONGER COMPOSITION that chains such a step several times (a multi-step jump);
  - a condition that reads SEVERAL fluents jointly, so a single move's effect depends on the
    wider state.
Recover whichever combination fits, as a rule for {self.target}, using at most one state
variable per clause."""


class DialMinter(IGGPMinter):
    """iggp_mint1 "dial": a cyclic counter with ``n_positions`` positions. Moves: ``inc`` (+1),
    ``dec`` (-1), ``stay`` and ``spin`` (+``spin_steps``). Intended rule (spin_steps=3):

        next_count(S,V) :- does(S,robot,A), action_inc(A),  true_count(S,V0), succ_count(V0,V).
        next_count(S,V) :- does(S,robot,A), action_dec(A),  true_count(S,V0), succ_count(V,V0).
        next_count(S,V) :- does(S,robot,A), action_stay(A), true_count(S,V).
        next_count(S,V) :- does(S,robot,A), action_spin(A), true_count(S,V0),
                           succ_count(V0,V1), succ_count(V1,V2), succ_count(V2,V).
    """

    problem_type = "iggp_dial"
    target = "next_count"
    actions = ["inc", "dec", "stay", "spin"]

    def __init__(self, out_root=None, n_positions: int = 6, spin_steps: int = 3, library_root=None):
        super().__init__(out_root, library_root)
        assert 1 < spin_steps < n_positions - 1, "spin must differ from inc, dec and stay"
        self.n_positions = n_positions
        self.spin_steps = spin_steps

    def max_vars(self):
        return 4 + self.spin_steps

    def max_body(self):
        return 3 + self.spin_steps

    def mint(self, seed, n_states: int = 160) -> ILPProblem:
        """One state per example: a random position and move, the correct next position as a
        positive and 2 wrong ones as negatives."""
        rng = random.Random(seed)
        n = self.n_positions
        steps = {"inc": 1, "dec": -1, "stay": 0, "spin": self.spin_steps}
        bk_facts = {
            "role": ["robot"],
            "my_input": [("robot", a) for a in self.actions],
            **{f"action_{a}": [a] for a in self.actions},
            **{f"count_{v}": [v] for v in range(n)},
            "succ_count": [(v, (v + 1) % n) for v in range(n)],
            "true_count": [],
            "does": [],
        }
        pos, neg = [], []
        for i in range(n_states):
            # state constants must not unify with dial values
            s = f"s{i}"
            v0 = rng.randrange(n)
            action = rng.choice(self.actions)
            v_correct = (v0 + steps[action]) % n
            bk_facts["true_count"].append((s, v0))
            bk_facts["does"].append((s, "robot", action))
            pos.append((s, v_correct))
            wrong = [v for v in range(n) if v != v_correct]
            neg += [(s, v) for v in rng.sample(wrong, k=min(2, len(wrong)))]
        return ILPProblem((self.target, 2), pos, neg, bk_facts)

    def get_glossary(self):
        last = self.n_positions - 1
        return f"""\
next_count(State, Value)   [TARGET] Value is the dial's position in the state that
                           follows State.
true_count(State, V0)      the dial currently reads V0 in State.
does(State, Agent, Action) Agent performed Action in State (Agent is always robot).
succ_count(V1, V2)         V2 is one step clockwise from V1 (cyclic: {last} -> 0).
role(Agent)                Agent is a player (here: robot).
my_input(Agent, Action)    Action is a legal move for Agent.
action_inc(Action)         Action advances the dial by one position.
action_dec(Action)         Action retreats the dial by one position.
action_stay(Action)        Action leaves the dial unchanged.
action_spin(Action)        A composite move advancing the dial by a fixed number of positions
                           greater than one.
count_0(V) ... count_{last}(V)  V is the named dial position 0 ... {last}."""


class SwitchesMinter(IGGPMinter):
    """iggp_mint2 "switches": ``n_switches`` boolean switches a, b, c, ...; the target is
    ``next_a(S)``, switch A being on after the move. Moves: ``hit_<x>`` toggles switch x, ``wait``
    does nothing, and two optional joint moves:

    - ``hit_all``: toggles every switch if they do not all agree, else turns all off;
    - ``sync``: sets every switch to the strict majority value (ties turn all off).

    Intended rule (n_switches=3, both joint moves):

        next_a(S) :- true_a(S),  does(S,robot,X), action_wait(X).
        next_a(S) :- true_a(S),  does(S,robot,X), action_hit_b(X).
        next_a(S) :- true_a(S),  does(S,robot,X), action_hit_c(X).
        next_a(S) :- false_a(S), does(S,robot,X), action_hit_a(X).
        next_a(S) :- false_a(S), true_b(S), does(S,robot,X), action_hit_all(X).
        next_a(S) :- false_a(S), true_c(S), does(S,robot,X), action_hit_all(X).
        next_a(S) :- true_a(S), true_b(S), does(S,robot,X), action_sync(X).
        next_a(S) :- true_a(S), true_c(S), does(S,robot,X), action_sync(X).
        next_a(S) :- true_b(S), true_c(S), does(S,robot,X), action_sync(X).

    ``hit_all`` needs n-1 clauses and ``sync`` one per majority-sized subset of switches.
    """

    problem_type = "iggp_switches"
    target = "next_a"

    def __init__(self, out_root=None, n_switches: int = 3, hit_all: bool = True, sync: bool = True,
                 library_root=None):
        super().__init__(out_root, library_root)
        assert 2 <= n_switches <= 26
        self.switches = list(string.ascii_lowercase[:n_switches])
        self.majority = n_switches // 2 + 1
        self.actions = [f"hit_{x}" for x in self.switches] + ["wait"]
        self.actions += ["hit_all"] * hit_all + ["sync"] * sync

    def next_value(self, state: dict[str, bool], action: str) -> bool:
        a = state["a"]
        if action == "hit_a":
            return not a
        if action == "hit_all":
            return not a if len(set(state.values())) > 1 else False
        if action == "sync":
            return sum(state.values()) >= self.majority
        return a

    def max_vars(self):
        return 3

    def max_body(self):
        return 2 + self.majority if "sync" in self.actions else 4

    def mint(self, seed, n_states: int = 160) -> ILPProblem:
        """One state per example, uniform switch values and move; positive iff switch A is on in
        the next state."""
        rng = random.Random(seed)
        bk_facts = {
            "role": ["robot"],
            "my_input": [("robot", a) for a in self.actions],
            **{f"action_{a}": [a] for a in self.actions},
            **{f"{v}_{x}": [] for x in self.switches for v in ("true", "false")},
            "does": [],
        }
        pos, neg = [], []
        for i in range(n_states):
            s = f"s{i}"
            state = {x: rng.random() < 0.5 for x in self.switches}
            action = rng.choice(self.actions)
            for x, on in state.items():
                bk_facts[f"{'true' if on else 'false'}_{x}"].append(s)
            bk_facts["does"].append((s, "robot", action))
            (pos if self.next_value(state, action) else neg).append(s)
        return ILPProblem((self.target, 1), pos, neg, bk_facts)

    def get_glossary(self):
        lines = ["next_a(State)              [TARGET] switch A is on in the state that follows State."]
        for x in self.switches:
            lines += [f"true_{x}(State) / false_{x}(State)",
                      f"                           switch {x.upper()} is currently on / off in State."]
        lines += [
            "does(State, Agent, Action) Agent performed Action in State (Agent is always robot).",
            "role(Agent)                Agent is a player (here: robot).",
            "my_input(Agent, Action)    Action is a legal move for Agent.",
        ]
        lines += [f"{f'action_hit_{x}(Action)':<27}Action flips switch {x.upper()} only." for x in self.switches]
        lines.append("action_wait(Action)        Action does nothing.")
        if "hit_all" in self.actions:
            lines += ["action_hit_all(Action)     A joint move over all switches whose effect depends on their",
                      "                           current values."]
        if "sync" in self.actions:
            lines += ["action_sync(Action)        A joint move that sets all switches according to a fixed",
                      "                           condition on their current values."]
        return "\n".join(lines)
