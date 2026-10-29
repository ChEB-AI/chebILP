"""IGGP tictactoe (Cropper, Evans & Law 2020), re-minted from a Python simulation of the GDL game.

The vocabulary follows the HuggingFace ``andrewcropper/ilp-datasets`` IGGP tasks: every state is an
example id ``e<i>`` and the first argument of each state relation (``true_cell``, ``true_control``,
``does``, ``does_mark``), constants are typed by unary predicates (``mark_x``, ``mypos_1``, ...),
and the GDL helpers ``row``, ``column``, ``diagonal``, ``line`` and ``open`` are not given. One
problem per IGGP target: ``goal``, ``terminal``, ``legal_mark``, ``next_cell`` and ``next_control``.
Each state contributes every candidate atom of the target, positive iff the GDL rules derive it.

States come from random playouts, as in IGGP: the player in control marks a random blank cell, the
other plays ``noop``. ``goal``, ``terminal`` and ``legal_mark`` use the distinct states visited,
terminal ones included; the ``next_*`` problems use distinct (non-terminal state, move) pairs.
"""

import random

from chebILP.ilp_problem import ILPProblem
from chebILP.minted_ilp_problems.base_minter import ILPProblemMinter

AGENTS = ("white", "black")
MARK_OF = {"white": "x", "black": "o"}
POSITIONS = (1, 2, 3)
CELLS = tuple((r, c) for r in POSITIONS for c in POSITIONS)
SCORES = (0, 50, 100)
LINES = (
    [[(r, c) for c in POSITIONS] for r in POSITIONS]
    + [[(r, c) for r in POSITIONS] for c in POSITIONS]
    + [[(1, 1), (2, 2), (3, 3)], [(1, 3), (2, 2), (3, 1)]]
)

# (max_vars, max_body) of the shortest rule over the base vocabulary: a line clause for goal and
# terminal; next_cell's blank frame rule needs an inequality the vocabulary lacks.
POPPER_BOUNDS = {
    "goal": (8, 9),
    "terminal": (6, 7),
    "legal_mark": (5, 3),
    "next_cell": (6, 5),
    "next_control": (3, 3),
}
ARITIES = {"goal": 3, "terminal": 1, "legal_mark": 4, "next_cell": 4, "next_control": 2}

# The GDL rules over this vocabulary, helpers included.
REFERENCE_PROGRAM = """\
state(E) :- true_control(E,_).
row(E,R,M) :- true_cell(E,R,1,M), true_cell(E,R,2,M), true_cell(E,R,3,M).
column(E,C,M) :- true_cell(E,1,C,M), true_cell(E,2,C,M), true_cell(E,3,C,M).
diagonal(E,M) :- true_cell(E,1,1,M), true_cell(E,2,2,M), true_cell(E,3,3,M).
diagonal(E,M) :- true_cell(E,1,3,M), true_cell(E,2,2,M), true_cell(E,3,1,M).
line(E,M) :- row(E,_,M).
line(E,M) :- column(E,_,M).
line(E,M) :- diagonal(E,M).
open(E) :- true_cell(E,_,_,b).

goal(E,white,100) :- line(E,x), not line(E,o).
goal(E,white,50) :- state(E), not line(E,x), not line(E,o).
goal(E,white,0) :- line(E,o), not line(E,x).
goal(E,black,100) :- line(E,o), not line(E,x).
goal(E,black,50) :- state(E), not line(E,x), not line(E,o).
goal(E,black,0) :- line(E,x), not line(E,o).

terminal(E) :- line(E,x).
terminal(E) :- line(E,o).
terminal(E) :- state(E), not open(E).

legal_mark(E,A,R,C) :- true_cell(E,R,C,b), true_control(E,A).

next_cell(E,R,C,x) :- does_mark(E,white,R,C), true_cell(E,R,C,b).
next_cell(E,R,C,o) :- does_mark(E,black,R,C), true_cell(E,R,C,b).
next_cell(E,R,C,M) :- true_cell(E,R,C,M), M != b.
next_cell(E,R,C,b) :- does_mark(E,_,J,_), true_cell(E,R,C,b), R != J.
next_cell(E,R,C,b) :- does_mark(E,_,_,K), true_cell(E,R,C,b), C != K.

next_control(E,white) :- true_control(E,black).
next_control(E,black) :- true_control(E,white).
"""


def has_line(board: dict, mark: str) -> bool:
    return any(all(board[cell] == mark for cell in line) for line in LINES)


def is_terminal(board: dict) -> bool:
    return has_line(board, "x") or has_line(board, "o") or "b" not in board.values()


def score(board: dict, agent: str) -> int | None:
    """The GDL goal value; ``None`` if both players have a line, which no reachable state has."""
    own, other = (
        has_line(board, MARK_OF[agent]),
        has_line(board, MARK_OF[AGENTS[agent == "white"]]),
    )
    if own and other:
        return None
    return 100 if own else 0 if other else 50


def playout(rng: random.Random) -> list:
    """One random game as ``[(board, control, move)]``; ``move`` is the marked cell, ``None`` in
    the final, terminal state."""
    board, control = {cell: "b" for cell in CELLS}, "white"
    steps = []
    while not is_terminal(board):
        move = rng.choice([cell for cell in CELLS if board[cell] == "b"])
        steps.append((dict(board), control, move))
        board[move] = MARK_OF[control]
        control = AGENTS[control == "white"]
    steps.append((dict(board), control, None))
    return steps


def _state_key(board: dict, control: str) -> tuple:
    return tuple(board[cell] for cell in CELLS), control


class TicTacToeMinter(ILPProblemMinter):
    """One IGGP tictactoe target; ``target`` is a key of ``ARITIES``."""

    problem_type = "iggp_tictactoe"

    def __init__(
        self,
        out_root=None,
        target: str = "goal",
        n_states: int = 500,
        library_root=None,
    ):
        super().__init__(out_root, library_root)
        assert target in ARITIES, f"unknown target {target}"
        self.target = target
        self.n_states = n_states

    def max_vars(self):
        return POPPER_BOUNDS[self.target][0]

    def max_body(self):
        return POPPER_BOUNDS[self.target][1]

    def static_predicates(self, ilp_problem):
        return self.state_free_predicates(ilp_problem)

    def sample_states(self, rng: random.Random, max_playouts: int = 100_000) -> list:
        """Distinct ``(board, control, move)`` from random playouts, in visit order. For the
        ``next_*`` targets a state is the (non-terminal state, move) pair, else the state alone
        with ``move`` None."""
        with_move = self.target.startswith("next_")
        seen, out = set(), []
        for _ in range(max_playouts):
            for board, control, move in playout(rng):
                if not with_move:
                    move = None
                elif move is None:
                    continue
                key = (_state_key(board, control), move if with_move else None)
                if key in seen:
                    continue
                seen.add(key)
                out.append((board, control, move))
                if len(out) == self.n_states:
                    return out
        raise RuntimeError(
            f"only {len(out)} distinct states after {max_playouts} playouts"
        )

    def candidate_atoms(
        self, board: dict, control: str, move
    ) -> list[tuple[tuple, bool]]:
        """Every candidate target atom of one state (without the state id), and whether it holds."""
        if self.target == "goal":
            return [((a, s), score(board, a) == s) for a in AGENTS for s in SCORES]
        if self.target == "terminal":
            return [((), is_terminal(board))]
        if self.target == "legal_mark":
            return [
                ((a, r, c), board[(r, c)] == "b" and control == a)
                for a in AGENTS
                for r, c in CELLS
            ]
        if self.target == "next_cell":
            after = dict(board)
            after[move] = MARK_OF[control]
            return [
                ((r, c, m), after[(r, c)] == m)
                for r, c in CELLS
                for m in ("b", "o", "x")
            ]
        if self.target == "next_control":
            return [((a,), a != control) for a in AGENTS]
        raise ValueError(self.target)

    def mint(self, seed, *args, **kwargs) -> ILPProblem:
        """State i is ``e<i>``; each contributes every candidate atom of the target."""
        rng = random.Random(seed)
        bk_facts = {
            "role": list(AGENTS),
            "index": list(POSITIONS),
            "input": [(a, "noop") for a in AGENTS],
            "input_mark": [(a, r, c) for a in AGENTS for r, c in CELLS],
            "agent_white": ["white"],
            "agent_black": ["black"],
            "mark_x": ["x"],
            "mark_o": ["o"],
            "mark_b": ["b"],
            **{f"mypos_{p}": [p] for p in POSITIONS},
            **{f"score_{s}": [s] for s in SCORES},
            "action_noop": ["noop"],
            "true_cell": [],
            "true_control": [],
            "does": [],
            "does_mark": [],
        }
        pos, neg = [], []
        for i, (board, control, move) in enumerate(self.sample_states(rng)):
            e = f"e{i}"
            bk_facts["true_cell"] += [(e, r, c, board[(r, c)]) for r, c in CELLS]
            bk_facts["true_control"].append((e, control))
            if move is not None:
                bk_facts["does_mark"].append((e, control, *move))
                bk_facts["does"].append((e, AGENTS[control == "white"], "noop"))
            for args, holds in self.candidate_atoms(board, control, move):
                (pos if holds else neg).append((e, *args) if args else e)
        return ILPProblem((self.target, ARITIES[self.target]), pos, neg, bk_facts)

    def get_task_description(self):
        return f"""\
IGGP (Inductive General Game Playing) turns a game written in the Game Description Language (GDL)
into ILP tasks. This task comes from TIC-TAC-TOE. Each example is one game STATE, named e0, e1,
...; it is the first argument of every state relation and of the target.

The game: two players, white and black, take turns on a 3x3 board, white first. Cells are
addressed by (row, column), each 1-3. A cell is blank (b) or holds a mark: white plays x, black
plays o. In each state one player is in control and marks a blank cell; the other player can only
play noop. A player who gets three of their marks in a row, column or diagonal wins. The game ends
when a player has three in a row or when no blank cell is left. Scores follow GDL: 100 for a win,
50 for a draw, 0 for a loss.

As in GDL, goal, terminal and legal are defined in every state, not only in final ones: goal gives
the score each player would receive in the state as it stands. The next_* relations describe the
state that follows the current one, given the moves taken in it.

Learn {self.target}/{ARITIES[self.target]} as rules over the background predicates, using one state
variable per rule."""

    def get_glossary(self):
        targets = {
            "goal": "goal(State, Agent, Score)   [TARGET] Agent's score in State is Score (0, 50 or 100).",
            "terminal": "terminal(State)             [TARGET] the game is over in State.",
            "legal_mark": "legal_mark(State, Agent, Row, Col)\n"
            "                            [TARGET] marking cell (Row, Col) is a legal move for Agent in State.",
            "next_cell": "next_cell(State, Row, Col, Mark)\n"
            "                            [TARGET] cell (Row, Col) holds Mark in the state that follows State.",
            "next_control": "next_control(State, Agent)  [TARGET] Agent is in control in the state that follows State.",
        }
        lines = [
            targets[self.target],
            "true_cell(State, Row, Col, Mark)",
            "                            cell (Row, Col) holds Mark (x, o or b for blank) in State.",
            "true_control(State, Agent)  Agent is the player in control in State.",
        ]
        if self.target.startswith("next_"):
            lines += [
                "does_mark(State, Agent, Row, Col)",
                "                            Agent marks cell (Row, Col) in State.",
                "does(State, Agent, Action)  Agent plays Action in State (here only noop).",
            ]
        lines += [
            "role(Agent)                 Agent is a player: white or black.",
            "index(Pos)                  Pos is a row or column index: 1, 2 or 3.",
            "input(Agent, Action)        Action (noop) is a possible move of Agent.",
            "input_mark(Agent, Row, Col) marking (Row, Col) is a possible move of Agent.",
            "agent_white(A) / agent_black(A)",
            "                            A is the constant white / black.",
            "mark_x(M) / mark_o(M) / mark_b(M)",
            "                            M is the constant x / o / b.",
            "mypos_1(P) / mypos_2(P) / mypos_3(P)",
            "                            P is the position 1 / 2 / 3.",
            "score_0(S) / score_50(S) / score_100(S)",
            "                            S is the score 0 / 50 / 100.",
            "action_noop(A)              A is the constant noop.",
        ]
        return "\n".join(lines)
