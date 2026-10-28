"""Game Description Language (GDL) games as clingo programs, with a batched simulator and IGGP-style
flattening.

A game file is KIF: ``(<= head body...)`` rules over ``role``, ``init``, ``true``, ``does``,
``next``, ``legal``, ``goal`` and ``terminal``. :class:`GDLGame` translates it to ASP. Every relation
that depends on ``true``/``does`` gets a leading state argument, so a single grounding answers
``legal``/``goal``/``terminal``/``next`` for a whole batch of states at once; the static relations
are grounded once.

Flattening follows the IGGP dataset (Cropper, Evans & Law 2020): fluent ``cell(1,2,x)`` becomes
``true_cell(1,2,x)``, move ``mark(1,2)`` by ``white`` becomes ``does_mark(white,1,2)`` (a constant
move ``noop`` stays ``does(white,noop)``), and the targets are ``next_<fluent>``,
``legal_<action>`` (``legal`` for constant moves), ``goal`` and ``terminal``.
"""
import itertools
import os
import re
from dataclasses import dataclass, field

import clingo

STATE_GUARD = "gdl_state"
# Relations that are dynamic whatever their bodies, and ones that must stay static.
FORCED_DYNAMIC = {"legal", "goal", "terminal", "next"}
STATIC_ONLY = {"role", "init", "base", "input"}
FLAT_PREFIXES = {"true", "next", "does", "legal", "input"}
PROLOG_PROTECTED = {"length", "member", "append", "last", "between", "number", "atom", "var", "nth0",
                    "nth1", "msort", "sort", "compare", "not", "is", "call", "integer", "atomic", "ground",
                    "functor", "arg", "findall", "forall", "true", "false", "halt", "write", "print"}
_TOKEN = re.compile(r"\(|\)|[^\s()]+")
_INT = re.compile(r"-?(0|[1-9]\d*)")
# A Prolog call: arguments with a comma or a variable. KIF may also write ``(init(cell 1 b))``.
_PROLOG_CALL = re.compile(r"\w\([^()]*(,|\b[A-Z_])")


# ---------------------------------------------------------------------------------------------
# KIF parsing

def _clean_lines(text: str, keep_percent: bool) -> str:
    """Drops ``;`` comments and Prolog lines (``:-`` or a Prolog call). ``%`` lines are dropped,
    or kept with the ``%`` prefix removed when ``keep_percent`` (some IGGP ``.pl`` files comment out
    all their KIF)."""
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%"):
            if not keep_percent:
                continue
            line = stripped.lstrip("%")
        line = line.split(";", 1)[0]
        if ":-" in line or _PROLOG_CALL.search(line):
            continue
        out.append(line)
    return "\n".join(out)


def parse_kif(text: str) -> list:
    """Top-level KIF forms as nested lists of strings.

    Skips stray Prolog: a token with a comma drops the form it occurs in, as does a fact with
    variables; bare top-level words and unmatched ``)`` are ignored.
    """
    forms, stack, bad = [], [], False
    for m in _TOKEN.finditer(text):
        tok = m.group()
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if not stack:
                continue
            form = stack.pop()
            if stack:
                # a nested ``(gameOver)`` is the constant gameOver
                stack[-1].append(form[0] if len(form) == 1 and isinstance(form[0], str) else form)
            else:
                if (not bad and len(form) >= 2 and isinstance(form[0], str) and not _INT.fullmatch(form[0])
                        and (form[0] == "<=" or "?" not in str(form))):
                    forms.append(form)
                bad = False
        elif stack:
            if "," in tok:
                bad = True
            stack[-1].append(tok)
    return forms


def read_game(path: str) -> list:
    """The KIF forms of a game file, falling back to its ``%``-commented lines if the live text
    declares no role."""
    with open(path) as f:
        text = f.read()
    forms = parse_kif(_clean_lines(text, keep_percent=False))
    if not any(isinstance(f, list) and f[0] == "role" for f in forms):
        forms = parse_kif(_clean_lines(text, keep_percent=True))
    return forms


# ---------------------------------------------------------------------------------------------
# Names and terms

def const(s: str) -> str:
    """A GDL constant as an ASP constant: integers stay integers, other names get non-word
    characters replaced and a ``c`` prefix unless they start lowercase."""
    if _INT.fullmatch(s):
        return s
    s = re.sub(r"\W", "_", s)
    return s if s[0].islower() else "c" + s


def _var(s: str) -> str:
    return "V_" + re.sub(r"\W", "_", s[1:])


def _is_var(x) -> bool:
    return isinstance(x, str) and x.startswith("?")


def _name(x) -> str:
    return x[0] if isinstance(x, list) else x


def _args(x) -> list:
    return x[1:] if isinstance(x, list) else []


def term(x) -> str:
    """An argument as an ASP term; compound GDL terms stay compound."""
    if isinstance(x, list):
        return f"{const(x[0])}({','.join(term(a) for a in x[1:])})"
    return _var(x) if _is_var(x) else const(x)


def flat_value(sym: clingo.Symbol):
    """One argument of a flattened atom: an int, a name, or a nested term joined by ``_``."""
    if sym.type == clingo.SymbolType.Number:
        return sym.number
    if sym.arguments:
        return "_".join([sym.name] + [str(flat_value(a)) for a in sym.arguments])
    return sym.name


def bk_relation(name: str) -> str:
    """A game relation's name in the flattened vocabulary: ISO builtins SWI-Prolog will not let a
    file redefine get ``my_``, names that could collide with a flattened ``true_*``/``next_*``/...
    predicate (firesheep's helper ``next_at``) get ``gdl_``."""
    if name in PROLOG_PROTECTED:
        return f"my_{name}"
    return f"gdl_{name}" if name.split("_", 1)[0] in FLAT_PREFIXES and "_" in name else name


def target_atoms(info: "StateInfo", target: str) -> set[tuple]:
    """The flattened positives of ``target`` (``goal``, ``terminal``, ``legal[_<move>]`` or
    ``next[_<fluent>]``) in one state, without the state argument."""
    if target == "goal":
        return {(flat_value(r), flat_value(v)) for r, v in info.goal}
    if target == "terminal":
        return {()} if info.terminal else set()
    if target.split("_", 1)[0] == "legal":
        atoms = (flat_atom("legal", m, (flat_value(r),)) for r, ms in info.legal.items() for m in ms)
    else:
        atoms = (flat_atom("next", f) for f in info.next)
    return {args for name, args in atoms if name == target}


def flat_atom(prefix: str, sym: clingo.Symbol, pre: tuple = ()) -> tuple[str, tuple]:
    """``true``/``next``/``does``/``legal``/``input`` over a fluent or move, as IGGP flattens them:
    ``cell(1,2,x)`` gives ``(<prefix>_cell, (*pre, 1, 2, x))``, a constant ``p`` gives
    ``(<prefix>, (*pre, p))``."""
    if sym.type == clingo.SymbolType.Function and sym.arguments:
        return f"{prefix}_{sym.name}", (*pre, *(flat_value(a) for a in sym.arguments))
    return prefix, (*pre, flat_value(sym))


# ---------------------------------------------------------------------------------------------
# Types (IGGP ``.typ`` files)

@dataclass
class GameTypes:
    """An IGGP type file: constants per type, subtypes, and the argument types of relations,
    fluents (``prop``) and moves (``action``)."""
    constants: dict = field(default_factory=dict)     # type -> [constant]
    subtypes: dict = field(default_factory=dict)      # type -> {types whose constants it includes}
    signatures: dict = field(default_factory=dict)    # name -> (arg types, kind)

    @classmethod
    def read(cls, path: str) -> "GameTypes":
        types = cls()
        with open(path) as f:
            text = f.read()
        for stmt in (s.strip() for s in text.split(".")):
            if ":>" in stmt:
                sub, sup = (s.strip() for s in stmt.split(":>"))
                types.subtypes.setdefault(sup, set()).add(sub)
            elif "::" in stmt:
                names, sig = stmt.split("::")
                parts = [p.strip() for p in sig.split("->")]
                for name in (n.strip() for n in names.split(",")):
                    if len(parts) == 1 and parts[0] not in ("prop", "bool"):
                        types.constants.setdefault(parts[0], []).append(const(name))
                    if len(parts) > 1 or parts[0] in ("prop", "bool", "action"):
                        types.signatures[const(name)] = (parts[:-1], parts[-1])
        return types

    def values(self, type_name: str) -> list:
        """Constants of a type and, transitively, of its subtypes, as flat values."""
        out, todo, seen = [], [type_name], set()
        while todo:
            t = todo.pop()
            if t in seen:
                continue
            seen.add(t)
            out += [int(c) if _INT.fullmatch(c) else c for c in self.constants.get(t, [])]
            todo += self.subtypes.get(t, ())
        return list(dict.fromkeys(out))

    def kind_names(self, kind: str, arity: int | None = None) -> list[str]:
        """Names declared with result ``kind`` (``prop``, ``action``), optionally of one arity."""
        return [n for n, (args, k) in self.signatures.items()
                if k == kind and n not in ("true", "next", "legal", "input", "does")
                and (arity is None or len(args) == arity)]


# ---------------------------------------------------------------------------------------------
# The game

@dataclass
class StateInfo:
    """What the game derives in one state; ``next`` only when the joint move was given."""
    legal: dict          # role -> [move symbols]
    goal: list           # [(role, value)]
    terminal: bool
    next: frozenset | None = None


class GDLGame:
    """One GDL game, translated to ASP."""

    def __init__(self, forms: list, name: str = "game"):
        self.name = name
        self.rules = [self._split(f) for f in forms]
        self.dynamic = self._dynamic_relations()
        static_src, dynamic_src = [], []
        for head, body in self.rules:
            if _name(head) in self.dynamic:
                dynamic_src += self._render(head, body)
            else:
                static_src += self._render(head, body)
        self.static_model = self._solve("\n".join(static_src))
        self.static_facts = "\n".join(f"{s}." for s in self.static_model)
        self.dynamic_program = "\n".join(dynamic_src)
        self.roles = [s.arguments[0] for s in self.static_model if s.name == "role" and len(s.arguments) == 1]
        self.init_state = frozenset(s.arguments[0] for s in self.static_model
                                    if s.name == "init" and len(s.arguments) == 1)

    @classmethod
    def from_file(cls, path: str) -> "GDLGame":
        return cls(read_game(path), os.path.splitext(os.path.basename(path))[0])

    @staticmethod
    def _split(form) -> tuple:
        if isinstance(form, list) and form[0] == "<=":
            return form[1], form[2:]
        return form, []

    def _dynamic_relations(self) -> set:
        """Relations depending on ``true``/``does``, plus the ones GDL evaluates per state."""
        dynamic = set(FORCED_DYNAMIC)
        changed = True
        while changed:
            changed = False
            for head, body in self.rules:
                h = _name(head)
                if h in dynamic or h in STATIC_ONLY:
                    continue
                if any(n in dynamic or n in ("true", "does") for n in self._body_relations(body)):
                    dynamic.add(h)
                    changed = True
        return dynamic

    def _body_relations(self, body) -> list[str]:
        out = []
        for lit in body:
            n = _name(lit)
            if n in ("not", "or"):
                out += self._body_relations(_args(lit))
            elif n != "distinct":
                out.append(n)
        return out

    # -- rendering ------------------------------------------------------------------------------

    def _atom(self, x, flat: bool) -> str:
        """A relation literal; dynamic relations get the state variable ``E``. In ``flat`` mode
        ``true``/``does``/``next``/``legal``/``input`` are flattened as in IGGP."""
        n, args = _name(x), _args(x)
        if flat and n in ("true", "next"):
            f = args[0]
            if _is_var(f):
                raise ValueError(f"cannot flatten {n} over a variable fluent")
            if not isinstance(f, list):
                return _flat_atom(n, ["E", const(f)])
            return _flat_atom(f"{n}_{const(f[0])}", ["E"] + [_flat_term(a) for a in f[1:]])
        if flat and n in ("does", "legal", "input"):
            r, a = args
            if _is_var(a):
                raise ValueError(f"cannot flatten {n} over a variable move")
            pre = [] if n == "input" else ["E"]
            if isinstance(a, list):
                return _flat_atom(f"{n}_{const(a[0])}", pre + [term(r)] + [_flat_term(t) for t in a[1:]])
            return _flat_atom(n, pre + [term(r), const(a)])
        terms = [_flat_term(a) if flat else term(a) for a in args]
        if n in self.dynamic or n in ("true", "does"):
            terms = ["E"] + terms
        name = bk_relation(const(n)) if flat else const(n)
        return f"{name}({','.join(terms)})" if terms else name

    def _literal_alternatives(self, lit, flat: bool, neg: bool = False) -> list[list[str]]:
        """A body literal as a disjunction of conjunctions; ``or`` splits, ``not`` pushes inward."""
        n = _name(lit)
        if n == "not":
            return self._literal_alternatives(lit[1], flat, not neg)
        if n == "or":
            alts = [self._literal_alternatives(a, flat, neg) for a in lit[1:]]
            if neg:  # not (a or b) = not a, not b
                return [sum(c, []) for c in itertools.product(*alts)]
            return [c for a in alts for c in a]
        if n == "distinct":
            a, b = (term(t) if not flat else _flat_term(t) for t in lit[1:])
            return [[f"{a}{'=' if neg else '!='}{b}"]]
        atom = self._atom(lit, flat)
        return [[f"not {atom}" if neg else atom]]

    def _render(self, head, body, flat: bool = False) -> list[str]:
        h = self._atom(head, flat)
        dynamic = _name(head) in self.dynamic
        alts = [self._literal_alternatives(lit, flat) for lit in body]
        rules = []
        for combo in itertools.product(*alts):
            lits = ([f"{STATE_GUARD}(E)"] if dynamic else []) + sum(combo, [])
            rules.append(f"{h} :- {', '.join(lits)}." if lits else f"{h}.")
        return rules

    def flat_program(self) -> str | None:
        """The dynamic rules in the flattened IGGP vocabulary, or None if a rule ranges over a
        variable fluent or move. ``gdl_state(E)`` must be supplied for every state."""
        out = []
        try:
            for head, body in self.rules:
                if _name(head) in self.dynamic:
                    out += self._render(head, body, flat=True)
        except ValueError:
            return None
        return "\n".join(out)

    # -- solving --------------------------------------------------------------------------------

    @staticmethod
    def _solve(program: str) -> list:
        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], program)
        ctl.ground([("base", [])])
        models = []
        ctl.solve(on_model=lambda m: models.append(m.symbols(atoms=True)))
        if not models:
            raise RuntimeError("the game program has no answer set")
        return sorted(models[0])

    def query(self, states: list, moves: list | None = None) -> list[StateInfo]:
        """``legal``/``goal``/``terminal`` per state, and ``next`` under ``moves[i]`` (a
        ``{role: move}`` dict, or None) when ``moves`` is given."""
        facts = []
        for i, state in enumerate(states):
            facts.append(f"{STATE_GUARD}({i}).")
            facts += [f"true({i},{f})." for f in state]
            if moves is not None and moves[i] is not None:
                facts += [f"does({i},{r},{a})." for r, a in moves[i].items()]
        model = self._solve("\n".join([self.dynamic_program, self.static_facts] + facts))
        infos = [StateInfo({r: [] for r in self.roles}, [], False,
                           frozenset() if moves is not None else None) for _ in states]
        nexts = [set() for _ in states]
        for s in model:
            if s.name not in ("legal", "goal", "terminal", "next") or not s.arguments:
                continue
            if s.arguments[0].type != clingo.SymbolType.Number:
                continue
            info, args = infos[s.arguments[0].number], s.arguments[1:]
            if s.name == "legal" and len(args) == 2:
                info.legal.setdefault(args[0], []).append(args[1])
            elif s.name == "goal" and len(args) == 2:
                info.goal.append((args[0], args[1]))
            elif s.name == "terminal" and not args:
                info.terminal = True
            elif s.name == "next" and len(args) == 1:
                nexts[s.arguments[0].number].add(args[0])
        if moves is not None:
            for info, nxt in zip(infos, nexts):
                info.next = frozenset(nxt)
        return infos

    # -- the static background, flattened -------------------------------------------------------

    def static_relations(self) -> dict[str, list[tuple]]:
        """Every static relation's extension as IGGP background: ``input`` split per move name,
        ``init``/``base`` and zero-arity facts dropped."""
        out = {}
        for s in self.static_model:
            if s.name in ("init", "base") or not s.arguments:
                continue
            if s.name == "input" and len(s.arguments) == 2:
                name, args = flat_atom("input", s.arguments[1], (flat_value(s.arguments[0]),))
            else:
                name, args = bk_relation(s.name), tuple(flat_value(a) for a in s.arguments)
            out.setdefault(name, []).append(args)
        return out

    # -- simulation -----------------------------------------------------------------------------

    def playouts(self, rng, n: int, max_steps: int = 200) -> list[list[tuple]]:
        """``n`` random playouts in lockstep: every role picks a uniformly random legal move.

        Each playout is ``[(state, joint move, StateInfo)]``; the info's ``next`` is the state the
        move leads to. The last step has move None, being terminal, stuck (a role without a legal
        move) or the ``max_steps`` cut-off.
        """
        traces = [[] for _ in range(n)]
        current = {i: self.init_state for i in range(n)}
        for step in range(max_steps + 1):
            if not current:
                break
            ids = sorted(current)
            infos = self.query([current[i] for i in ids])
            moving, moves = [], []
            for i, info in zip(ids, infos):
                options = [sorted(info.legal.get(r, []), key=str) for r in self.roles]
                if info.terminal or step == max_steps or not all(options):
                    traces[i].append((current[i], None, info))
                    del current[i]
                    continue
                moving.append(i)
                moves.append({r: rng.choice(opts) for r, opts in zip(self.roles, options)})
            if not moving:
                break
            by_id = dict(zip(ids, infos))
            for i, move, after in zip(moving, moves, self.query([current[i] for i in moving], moves)):
                by_id[i].next = after.next
                traces[i].append((current[i], move, by_id[i]))
                current[i] = after.next
        return traces


def _flat_term(x) -> str:
    """An argument in flat mode: compound ground terms become joined constants."""
    if isinstance(x, list):
        if any(_is_var(a) or isinstance(a, list) for a in x[1:]):
            raise ValueError("cannot flatten a compound argument with variables")
        return "_".join(const(a) for a in x)
    return term(x)


def _flat_atom(name: str, terms: list[str]) -> str:
    return f"{name}({','.join(terms)})"
