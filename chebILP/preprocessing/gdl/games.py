"""GDL games of the IGGP repository and their targets, in the IGGP vocabulary (Cropper, Evans & Law 2020).

Games and types come from the IGGP repository (``git clone https://github.com/andrewcropper/iggp
data/iggp``): ``games/<game>.txt`` (or ``.pl``) holds the GDL, ``types/<game>.typ`` the argument
types. Each game has the targets ``goal``, ``terminal``, ``legal[_<move>]`` and ``next[_<fluent>]``.

States come from random playouts, simulated with :class:`GDLGame`. :class:`GameTarget` gives a
target's candidate atoms (every atom it could have in a state), its glossary and its reference
program.
"""

import functools
import itertools
import os
import random

from chebILP.preprocessing.gdl.gdl import (
    STATE_GUARD,
    GameTypes,
    GDLGame,
    const,
    flat_atom,
    flat_value,
    parse_kif,
    read_game,
)

IGGP_ROOT = os.path.join("data", "iggp")
# Popper reserves pos/1.
TYPE_RENAMES = {"pos": "mypos"}
# KIF the repository's game files lost to Prolog edits.
GAME_PATCHES = {
    "fizzbuzz": "(legal player (say fizz)) (legal player (say buzz)) (legal player (say fizzbuzz))",
}


def game_paths(game: str, root: str = IGGP_ROOT) -> tuple[str | None, str | None]:
    """The game's GDL file (``.txt`` preferred over ``.pl``) and type file, None where missing."""
    gdl = next(
        (
            p
            for ext in ("txt", "pl")
            if os.path.exists(p := os.path.join(root, "games", f"{game}.{ext}"))
        ),
        None,
    )
    typ = os.path.join(root, "types", f"{game}.typ")
    return gdl, typ if os.path.exists(typ) else None


def iggp_games(root: str = IGGP_ROOT) -> list[str]:
    """The 50 games of the IGGP dataset (those with ``data/train`` examples); empty if not cloned."""
    train = os.path.join(root, "data", "train")
    if not os.path.isdir(train):
        return []
    return sorted(
        {f.rsplit("_", 2)[0] for f in os.listdir(train) if f.endswith("_train.dat")}
    )


@functools.cache
def load_game(game: str, root: str = IGGP_ROOT) -> GDLGame:
    gdl, _ = game_paths(game, root)
    if gdl is None:
        raise FileNotFoundError(f"no GDL file for {game} under {root}/games")
    return GDLGame(read_game(gdl) + parse_kif(GAME_PATCHES.get(game, "")), game)


@functools.cache
def load_types(game: str, root: str = IGGP_ROOT) -> GameTypes | None:
    _, typ = game_paths(game, root)
    return GameTypes.read(typ) if typ else None


@functools.cache
def gdl_targets(game: str, root: str = IGGP_ROOT) -> list[str]:
    """The game's targets: ``goal``, ``terminal``, then one ``legal[_<move>]`` per move name in
    ``input`` and one ``next[_<fluent>]`` per fluent name in ``base`` or ``init``. Empty if the
    game's files are missing."""
    if game_paths(game, root)[0] is None:
        return []
    g = load_game(game, root)
    return ["goal", "terminal"] + list(shapes(g, "legal")) + list(shapes(g, "next"))


def sample_steps(
    game: str,
    root: str,
    seed: int,
    with_moves: bool,
    n_states: int,
    max_steps: int,
    max_playouts: int = 20_000,
) -> tuple:
    """Distinct ``(state, joint move, StateInfo)`` from random playouts, in playout order: the
    (state, move) pair when ``with_moves``, skipping final states, else the state with move None.

    Playouts run in batches that double from 4 to 64. Stops early, with fewer than ``n_states``,
    once a batch of 64 finds nothing new.
    """
    rng, g = random.Random(seed), load_game(game, root)
    seen, out, batch, done = set(), [], 4, 0
    while done < max_playouts:
        found = False
        for trace in g.playouts(rng, batch, max_steps):
            for state, move, info in trace:
                if with_moves and move is None:
                    continue
                move = move if with_moves else None
                key = (state, frozenset(move.items()) if move else None)
                if key in seen:
                    continue
                seen.add(key)
                found = True
                out.append((state, move, info))
                if len(out) == n_states:
                    return tuple(out)
        done += batch
        if not found and batch == 64:
            break
        batch = min(2 * batch, 64)
    print(f"[{game}] only {len(out)} distinct examples")
    return tuple(out)


def typing_predicates(game: str, root: str = IGGP_ROOT) -> dict[str, list]:
    """``<type>_<constant>(c)`` for every constant the game's type file declares."""
    types = load_types(game, root)
    out = {}
    for type_name, consts in types.constants.items() if types else ():
        for c in consts:
            v = int(c) if c.lstrip("-").isdigit() else c
            out[
                f"{TYPE_RENAMES.get(type_name, type_name)}_{str(c).replace('-', 'm')}"
            ] = [v]
    return out


class GameTarget:
    """One target of one GDL game."""

    def __init__(self, game: str, target: str, root: str = IGGP_ROOT):
        self.game = game
        self.target = target
        self.root = root

    @property
    def gdl(self) -> GDLGame:
        return load_game(self.game, self.root)

    @property
    def types(self) -> GameTypes | None:
        return load_types(self.game, self.root)

    @property
    def with_moves(self) -> bool:
        """Whether the examples are (state, joint move) pairs, as for the ``next`` targets."""
        return self.target.split("_", 1)[0] == "next"

    # -- candidates -----------------------------------------------------------------------------

    def _arg_types(self) -> list[str] | None:
        """The target's argument types from the type file, None if it does not declare them."""
        types, t = self.types, self.target
        if types is None:
            return None

        def sig(n):
            return types.signatures.get(n, (None,))[0]

        if t == "terminal":
            return []
        if t == "goal":
            return sig("goal")
        kind, _, name = t.partition("_")
        if kind == "legal":
            agent = (sig("legal") or ["agent"])[0]
            if not name:
                return [agent, "<constant moves>"]
            return None if sig(name) is None else [agent] + sig(name)
        if not name:
            return ["<constant fluents>"]
        return sig(name)

    def _type_values(self, type_name: str) -> list:
        if type_name == "<constant moves>":
            return self.types.values("action")
        if type_name == "<constant fluents>":
            return [
                n
                for n, (args, kind) in self.types.signatures.items()
                if kind == "prop" and not args
            ]
        return self.types.values(type_name)

    def _declared(self) -> list[tuple] | None:
        """The target's atoms as the game declares them: ``base`` fluents for ``next``, ``input``
        moves for ``legal``, every role with every value a ``goal`` head writes. None if the game
        declares none (or a goal head leaves its value open)."""
        g, t = self.gdl, self.target
        kind = t.split("_", 1)[0]
        if t == "goal":
            values = [
                h[2]
                for h, _ in g.rules
                if isinstance(h, list) and h[0] == "goal" and len(h) == 3
            ]
            if not values or any(
                isinstance(v, list) or v.startswith("?") for v in values
            ):
                return None
            values = sorted(
                {int(v) if v.lstrip("-").isdigit() else const(v) for v in values},
                key=str,
            )
            return [(flat_value(r), v) for r in g.roles for v in values]
        if kind == "next":
            atoms = [
                flat_atom("next", s.arguments[0])
                for s in g.static_model
                if s.name == "base" and len(s.arguments) == 1
            ]
        elif kind == "legal":
            atoms = [
                flat_atom("legal", s.arguments[1], (flat_value(s.arguments[0]),))
                for s in g.static_model
                if s.name == "input" and len(s.arguments) == 2
            ]
        else:
            return None
        return [args for name, args in atoms if name == t] or None

    def candidates(self, positives: list[set]) -> list[tuple]:
        """Every possible target atom (without the state): as the game declares them, else the
        type-correct ones, else the union of observed positives. Observed positives are always
        included."""
        observed = set().union(*positives)
        out, arg_types = self._declared(), self._arg_types()
        if out is None and arg_types is not None:
            out = list(itertools.product(*(self._type_values(t) for t in arg_types)))
        if out is None:
            print(
                f"[{self.game}/{self.target}] neither declared nor typed, candidates are the observed positives"
            )
            return sorted(observed, key=str)
        extra = observed - set(out)
        if extra:
            print(
                f"[{self.game}/{self.target}] {len(extra)} positive(s) outside the candidates, e.g. {next(iter(extra))}"
            )
        return list(dict.fromkeys(out)) + sorted(extra, key=str)

    def reference_program(self, head: str) -> str | None:
        """The game's dynamic rules in this vocabulary, the IGGP ground truth with its helper
        relations, defining ``head`` as the target; None where a rule ranges over a variable
        fluent or move."""
        prog = self.gdl.flat_program()
        if prog is None:
            return None
        guards = [
            f"{STATE_GUARD}(E) :- {name}(E{',_' * n})."
            for name, n in shapes(self.gdl, "true").items()
        ]
        args = ",".join(f"V{i}" for i in range(len(self.head_args())))
        return "\n".join([prog, *guards, f"{head}({args}) :- {self.target}({args})."])

    # -- prompt text ----------------------------------------------------------------------------

    def _arg_names(self, name: str, n: int) -> list[str]:
        """Argument names of a relation, fluent or move from its type signature, else ``A1``..."""
        ts = self.types.signatures.get(name, (None,))[0] if self.types else None
        return (
            [t.capitalize() for t in ts]
            if ts and len(ts) == n
            else [f"A{i + 1}" for i in range(n)]
        )

    def head_args(self) -> list[str]:
        """Distinct argument names of the target, the state first (``Pos, Pos2``)."""
        t = self.target
        kind, _, name = t.partition("_")
        if t == "goal":
            args = ["State", *self._arg_names("goal", 2)]
        elif t == "terminal":
            args = ["State"]
        elif kind == "legal":
            n = shapes(self.gdl, "legal").get(t, 2) - 1
            args = ["State", "Agent", *(self._arg_names(name, n) if name else ["Move"])]
        else:
            n = shapes(self.gdl, "next").get(t, 1)
            args = ["State", *(self._arg_names(name, n) if name else ["F"])]
        out = []
        for a in args:
            name, k = a, 1
            # a renamed duplicate avoids every original name, so none is taken twice
            while name in out or (k > 1 and name in args):
                k += 1
                name = f"{a}{k}"
            out.append(name)
        return out

    def target_text(self) -> str:
        """What the target says, over the names of :meth:`head_args`."""
        t = self.target
        kind, _, name = t.partition("_")
        if t == "goal":
            return "the player's goal value in State"
        if t == "terminal":
            return "the game is over in State"
        if kind == "legal":
            return f"Agent may play {_what(name, 'move Move')} in State"
        return f"{_what(name, 'fluent F')} holds in the state that follows State"

    def description(self, head: str) -> str:
        """The class definition shown in prompts."""
        moves = " and the move each player takes in it" if self.with_moves else ""
        return (
            f"Target of the game {self.game.replace('_', ' ').upper()}: "
            f"{head}({', '.join(self.head_args())}) holds when {self.target_text()}. Each example "
            f"is one state of a random playout, given by the fluents true in it{moves}."
        )

    def glossary(self, head: str) -> str:
        """The background predicates of the target's problems, and the target as ``head``."""
        lines = [_entry(self.head_args(), head, f"[TARGET] {self.target_text()}.")]
        for pred, n in shapes(self.gdl, "true").items():
            f = pred[5:]
            lines.append(
                _entry(
                    ["State", *(self._arg_names(f, n) if f else ["F"])],
                    pred,
                    f"{_what(f, 'fluent F')} holds in State.",
                )
            )
        if self.with_moves:
            for pred, n in shapes(self.gdl, "does").items():
                m = pred[5:]
                lines.append(
                    _entry(
                        [
                            "State",
                            "Agent",
                            *(self._arg_names(m, n - 1) if m else ["Move"]),
                        ],
                        pred,
                        f"Agent plays {_what(m, 'move Move')} in State.",
                    )
                )
        for pred, facts in self.gdl.static_relations().items():
            n = len(facts[0])
            if pred == "role":
                lines.append(_entry(["Agent"], pred, "Agent is a player."))
            elif pred.split("_", 1)[0] == "input":
                m = pred[6:]
                lines.append(
                    _entry(
                        ["Agent", *(self._arg_names(m, n - 1) if m else ["Move"])],
                        pred,
                        f"{_what(m, 'move Move')} is a possible move of Agent.",
                    )
                )
            else:
                lines.append(
                    _entry(
                        self._arg_names(pred.removeprefix("my_"), n),
                        pred,
                        "a static relation of the game.",
                    )
                )
        for type_name, consts in self.types.constants.items() if self.types else ():
            names = [
                f"{TYPE_RENAMES.get(type_name, type_name)}_{str(c).replace('-', 'm')}(X)"
                for c in consts
            ]
            if len(consts) <= 4:
                lines.append(
                    _line(
                        " / ".join(names),
                        f"X is the {type_name} {' / '.join(map(str, consts))}.",
                    )
                )
            else:
                lines.append(
                    _line(
                        f"{names[0]} ... {names[-1]}",
                        f"X is the {type_name} {consts[0]} ... {consts[-1]}.",
                    )
                )
        return "\n".join(lines)


def shapes(game: GDLGame, prefix: str) -> dict[str, int]:
    """Flattened predicate name -> argument count (without the state) for ``true``/``next`` over
    the fluents in ``base``, ``init`` and ``next`` heads, or ``does``/``legal``/``input`` over the
    moves in ``input`` and ``legal`` heads."""
    fluents = prefix in ("true", "next")
    if fluents:
        syms = [
            (s.arguments[0], ())
            for s in game.static_model
            if s.name in ("base", "init") and len(s.arguments) == 1
        ]
    else:
        syms = [
            (s.arguments[1], (s.arguments[0],))
            for s in game.static_model
            if s.name == "input" and len(s.arguments) == 2
        ]
    out = {n: len(a) for n, a in (flat_atom(prefix, m, pre) for m, pre in syms)}
    for head, _ in game.rules:
        if not isinstance(head, list) or head[0] != ("next" if fluents else "legal"):
            continue
        x = head[-1]
        if isinstance(x, list):
            out.setdefault(f"{prefix}_{const(x[0])}", len(x) - 1 + (not fluents))
        elif not x.startswith("?"):
            out.setdefault(prefix, 1 + (not fluents))
    return dict(sorted(out.items()))


def _what(name: str, kind: str) -> str:
    """``name(...)``, or for the constant form ``the constant <kind>`` (``move Move``, ``fluent F``)."""
    return f"{name}(...)" if name else f"the constant {kind}"


def _entry(args: list[str], name: str, text: str) -> str:
    return _line(f"{name}({', '.join(args)})", text)


def _line(head: str, text: str) -> str:
    return head.ljust(32) + text if len(head) < 32 else f"{head}\n{'':32}{text}"
