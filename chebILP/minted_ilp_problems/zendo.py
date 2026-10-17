"""Zendo-style minted problems (Hocquette & Cropper, AAAI 2023).

Each example is one scene ("state") of coloured, sized, oriented pieces, some touching one
another; the target ``zendo(State)`` holds iff the scene satisfies a hidden rule. All targets
here are novel (none restates a canonical zendo1..zendo10 rule). Scenes are rejection-sampled
with forced positives, near-miss negatives and decoys planted in both classes, so that a
partial pattern is not itself discriminative.
"""
import random
from dataclasses import dataclass

from chebILP.ilp_problem import ILPProblem
from chebILP.minted_ilp_problems.base_minter import ILPProblemMinter

COLORS = ("blue", "green", "red")
ORIENTATIONS = ("upright", "lhs", "rhs", "strange")
TILTED = ("lhs", "rhs", "strange")
SIZE_MIN, SIZE_MAX = 0, 10
SMALL_VALS = (0, 1, 2, 3)
MEDIUM_VALS = (4, 5, 6)
LARGE_VALS = (7, 8, 9, 10)
COORD_MIN, COORD_MAX = 0, 20


@dataclass
class Scene:
    n_pieces: int
    color: list
    orientation: list
    size: list
    coord1: list
    coord2: list
    contacts: set  # frozenset({i, j}) per touching pair

    def touches(self, i: int) -> list:
        out = []
        for pair in self.contacts:
            if i in pair:
                (j,) = pair - {i}
                out.append(j)
        return out


def size_bucket(v: int) -> str:
    if v in SMALL_VALS:
        return "small"
    if v in MEDIUM_VALS:
        return "medium"
    return "large"


def random_scene(rng: random.Random, n_pieces: int, contact_prob: float = 0.45) -> Scene:
    color = [rng.choice(COLORS) for _ in range(n_pieces)]
    orientation = [rng.choice(ORIENTATIONS) for _ in range(n_pieces)]
    size = [rng.randint(SIZE_MIN, SIZE_MAX) for _ in range(n_pieces)]
    coord1 = [rng.randint(COORD_MIN, COORD_MAX) for _ in range(n_pieces)]
    coord2 = [rng.randint(COORD_MIN, COORD_MAX) for _ in range(n_pieces)]
    contacts = set()
    for i in range(n_pieces):
        for j in range(i + 1, n_pieces):
            if rng.random() < contact_prob:
                contacts.add(frozenset((i, j)))
    return Scene(n_pieces, color, orientation, size, coord1, coord2, contacts)


def force_size_chain(rng: random.Random, sc: Scene) -> tuple:
    """Makes three random pieces a small -> medium -> large contact chain."""
    a, b, c = rng.sample(range(sc.n_pieces), 3)
    sc.size[a] = rng.choice(SMALL_VALS)
    sc.size[b] = rng.choice(MEDIUM_VALS)
    sc.size[c] = rng.choice(LARGE_VALS)
    sc.contacts.add(frozenset((a, b)))
    sc.contacts.add(frozenset((b, c)))
    return a, b, c


def has_size_chain(sc: Scene, top=lambda c: True, same_color=False, root=lambda a: True) -> bool:
    """A small -> medium -> large contact chain whose root and top satisfy the given conditions."""
    for a in range(sc.n_pieces):
        if size_bucket(sc.size[a]) != "small" or not root(a):
            continue
        for b in sc.touches(a):
            if size_bucket(sc.size[b]) != "medium" or (same_color and sc.color[b] != sc.color[a]):
                continue
            for c in sc.touches(b):
                if (size_bucket(sc.size[c]) == "large" and top(c)
                        and (not same_color or sc.color[c] == sc.color[a])):
                    return True
    return False


class ZendoMinter(ILPProblemMinter):
    """Shared scene sampler, vocabulary and leakage-free description. A subclass defines the
    hidden rule through ``check`` and optionally shapes the sampling through ``force``,
    ``neg_force`` and ``plant_decoy``."""

    problem_type = "zendo"
    target = "zendo"
    _max_vars: int
    _max_body: int

    def max_vars(self):
        return self._max_vars

    def max_body(self):
        return self._max_body

    def check(self, sc: Scene) -> bool:
        """The ground-truth label."""
        raise NotImplementedError

    def force(self, rng: random.Random, sc: Scene) -> tuple | None:
        """Edits the scene so the pattern holds; returns the pieces it reserved."""
        raise NotImplementedError

    def neg_force(self, rng: random.Random, sc: Scene, indices: tuple | None) -> None:
        """Breaks a forced pattern in one place only, giving a near-miss negative."""

    def plant_decoy(self, rng: random.Random, sc: Scene, reserved: set) -> None:
        """Plants a strong partial pattern outside ``reserved``, in scenes of both classes."""

    has_neg_force = False

    def sample_scenes(self, rng: random.Random, n_pos: int, n_neg: int, max_attempts: int = 500_000):
        pos, neg = [], []
        attempts = 0
        while (len(pos) < n_pos or len(neg) < n_neg) and attempts < max_attempts:
            attempts += 1
            sc = random_scene(rng, rng.randint(3, 6))
            indices = None
            if self.has_neg_force and len(neg) < n_neg and rng.random() < 0.5:
                indices = self.force(rng, sc)
                self.neg_force(rng, sc, indices)
            elif rng.random() < 0.5:
                indices = self.force(rng, sc)
            self.plant_decoy(rng, sc, set(indices) if indices else set())
            label = self.check(sc)
            if label and len(pos) < n_pos:
                pos.append(sc)
            elif not label and len(neg) < n_neg:
                neg.append(sc)
        if len(pos) < n_pos or len(neg) < n_neg:
            raise RuntimeError(f"{type(self).__name__}: only got {len(pos)} pos / {len(neg)} neg "
                               f"after {attempts} attempts")
        return pos, neg

    def mint(self, seed, n_pos: int = 50, n_neg: int = 50) -> ILPProblem:
        """States 0..n_pos-1 are positive, the rest negative; piece i of state s is ``p<s>_<i>``."""
        rng = random.Random(seed)
        pos, neg = self.sample_scenes(rng, n_pos, n_neg)
        scenes = list(enumerate(pos + neg))
        pieces = [(s, f"p{s}_{i}", i, sc) for s, sc in scenes for i in range(sc.n_pieces)]
        contacts = []
        for s, sc in scenes:
            for i, j in sorted(tuple(sorted(pair)) for pair in sc.contacts):
                contacts += [(f"p{s}_{i}", f"p{s}_{j}"), (f"p{s}_{j}", f"p{s}_{i}")]
        bk_facts = {
            **{c: [p for _, p, i, sc in pieces if sc.color[i] == c] for c in COLORS},
            **{o: [p for _, p, i, sc in pieces if sc.orientation[i] == o] for o in ORIENTATIONS},
            "contact": contacts,
            "coord1": [(p, sc.coord1[i]) for _, p, i, sc in pieces],
            "coord2": [(p, sc.coord2[i]) for _, p, i, sc in pieces],
            "max_size": [SIZE_MAX],
            "small": list(SMALL_VALS),
            "medium": list(MEDIUM_VALS),
            "large": list(LARGE_VALS),
            "size": [(p, sc.size[i]) for _, p, i, sc in pieces],
            "piece": [(s, p) for s, p, _, _ in pieces],
        }
        return ILPProblem((self.target, 1), list(range(n_pos)), list(range(n_pos, n_pos + n_neg)), bk_facts)

    def get_task_description(self):
        return f"""\
Zendo is an inductive-reasoning game: a hidden rule classifies structures (called "states")
as either following the rule or not. Each state contains one or more pieces (piece/2 links a
state to its pieces). Every piece has a colour (blue, green or red), a size (small, medium or
large -- given indirectly via an integer size value classified by the size predicates), an
orientation (upright, lhs, rhs or strange), and two spatial coordinates (coord1, coord2).
Pieces may be in contact with one another (contact/2, symmetric, within the same state).

The hidden rule for {self.target}(State) is a single structural pattern over the pieces of a
state and how they relate.
Recover whichever pattern separates the positive from the negative states, and express it as
a rule for {self.target}(State), using at most one state variable per rule."""

    def get_glossary(self):
        return """\
zendo(State)              [TARGET] the state satisfies the hidden Zendo rule.
piece(State, Piece)       Piece belongs to State.
contact(Piece1, Piece2)   Piece1 is touching Piece2 (same state).
coord1(Piece, Real)       first (x) coordinate of Piece.
coord2(Piece, Real)       second (y) coordinate of Piece.
size(Piece, Real)         the size value of Piece (classified by small/medium/large below).
max_size(Real)            the largest possible size value.
small(Real)               the size value is small (likewise medium/large).
medium(Real) / large(Real)
blue(Piece)               Piece is blue (likewise green, red).
green(Piece) / red(Piece)
upright(Piece)            Piece stands upright (likewise lhs, rhs, strange orientations).
lhs(Piece) / rhs(Piece) / strange(Piece)"""


class TiltedChainMinter(ZendoMinter):
    """zendo_mint1: a small piece touches a medium piece that touches a large, tilted piece.

        zendo(S) :- piece(S,A), size(A,SA), small(SA), contact(A,B), size(B,SB), medium(SB),
                    contact(B,C), size(C,SC), large(SC), not upright(C).

    Near misses make the top piece upright; decoys give another piece the bare large+tilted
    conjunction, which would otherwise proxy the whole chain.
    """

    _max_vars, _max_body = 7, 10
    has_neg_force = True

    def force(self, rng, sc):
        a, b, c = force_size_chain(rng, sc)
        sc.orientation[c] = rng.choice(TILTED)
        return a, b, c

    def neg_force(self, rng, sc, indices):
        sc.orientation[indices[2]] = "upright"

    def plant_decoy(self, rng, sc, reserved):
        candidates = [i for i in range(sc.n_pieces) if i not in reserved]
        if not candidates or rng.random() >= 0.8:
            return
        d = rng.choice(candidates)
        sc.size[d] = rng.choice(LARGE_VALS)
        sc.orientation[d] = rng.choice(TILTED)

    def check(self, sc):
        return has_size_chain(sc, top=lambda c: sc.orientation[c] != "upright")


class RedChainMinter(ZendoMinter):
    """zendo_mint2: a red small piece touches a medium piece that touches a large piece.

        zendo(S) :- piece(S,A), size(A,SA), small(SA), red(A), contact(A,B), size(B,SB),
                    medium(SB), contact(B,C), size(C,SC), large(SC).

    Near misses recolour the root; decoys plant a small red piece touching a medium piece with
    no large continuation, so the two-tier partial is not discriminative.
    """

    _max_vars, _max_body = 7, 10
    has_neg_force = True

    def force(self, rng, sc):
        a, b, c = force_size_chain(rng, sc)
        sc.color[a] = "red"
        return a, b, c

    def neg_force(self, rng, sc, indices):
        sc.color[indices[0]] = rng.choice(("blue", "green"))

    def plant_decoy(self, rng, sc, reserved):
        candidates = [i for i in range(sc.n_pieces) if i not in reserved]
        if rng.random() >= 0.9:
            return
        if len(candidates) >= 2:
            d, e = rng.sample(candidates, 2)
            sc.size[d] = rng.choice(SMALL_VALS)
            sc.color[d] = "red"
            sc.size[e] = rng.choice(MEDIUM_VALS)
            sc.contacts.add(frozenset((d, e)))
            # the planted medium must touch no large piece, or the decoy completes a chain
            for f in sc.touches(e):
                if f != d and f not in reserved and size_bucket(sc.size[f]) == "large":
                    sc.size[f] = rng.choice(MEDIUM_VALS)
        elif candidates:
            d = candidates[0]
            sc.size[d] = rng.choice(SMALL_VALS)
            sc.color[d] = "red"

    def check(self, sc):
        return has_size_chain(sc, root=lambda a: sc.color[a] == "red")


class ChainMinter(ZendoMinter):
    """zendo_mint3: a small piece touches a medium piece that touches a large piece.

        zendo(S) :- piece(S,A), size(A,SA), small(SA), contact(A,B), size(B,SB),
                    medium(SB), contact(B,C), size(C,SC), large(SC).
    """

    _max_vars, _max_body = 7, 9

    def force(self, rng, sc):
        force_size_chain(rng, sc)

    def check(self, sc):
        return has_size_chain(sc)


class Ascend4Minter(ZendoMinter):
    """zendo_mint4: a simple contact path A-B-C-D of four pieces with strictly increasing size
    values, size(A) < size(B) < size(C) < size(D). Needs a size comparison chained three times.

    Near misses break the last hop; decoys plant a three-piece ascending path.
    """

    _max_vars, _max_body = 9, 12
    has_neg_force = True

    def force(self, rng, sc):
        if sc.n_pieces < 4:
            return None
        a, b, c, d = rng.sample(range(sc.n_pieces), 4)
        sc.size[a], sc.size[b], sc.size[c], sc.size[d] = sorted(rng.sample(range(SIZE_MIN, SIZE_MAX + 1), 4))
        sc.contacts.add(frozenset((a, b)))
        sc.contacts.add(frozenset((b, c)))
        sc.contacts.add(frozenset((c, d)))
        return a, b, c, d

    def neg_force(self, rng, sc, indices):
        if indices is None:
            return
        _a, _b, c, d = indices
        sc.size[d] = rng.randint(SIZE_MIN, sc.size[c])

    def plant_decoy(self, rng, sc, reserved):
        candidates = [i for i in range(sc.n_pieces) if i not in reserved]
        if len(candidates) < 3 or rng.random() >= 0.85:
            return
        p, q, r = rng.sample(candidates, 3)
        sc.size[p], sc.size[q], sc.size[r] = sorted(rng.sample(range(SIZE_MIN, SIZE_MAX + 1), 3))
        sc.contacts.add(frozenset((p, q)))
        sc.contacts.add(frozenset((q, r)))
        for f in sc.touches(r):  # no fourth ascending hop off the top
            if f not in (p, q) and f not in reserved and sc.size[f] > sc.size[r]:
                sc.size[f] = rng.randint(SIZE_MIN, sc.size[r])

    def check(self, sc):
        def dfs(path):
            if len(path) == 4:
                return True
            last = path[-1]
            return any(nxt not in path and sc.size[nxt] > sc.size[last] and dfs(path + [nxt])
                       for nxt in sc.touches(last))
        return any(dfs([start]) for start in range(sc.n_pieces))


class RedTiltedTopMinter(ZendoMinter):
    """zendo_mint5: a small -> medium -> large contact chain whose large top piece is both red
    and tilted.

        zendo(S) :- piece(S,A), size(A,SA), small(SA), contact(A,B), size(B,SB), medium(SB),
                    contact(B,C), size(C,SC), large(SC), red(C), not upright(C).

    Near misses keep one of the two top anchors; decoys plant a medium piece touching a large
    red tilted piece with no small root.
    """

    _max_vars, _max_body = 8, 11
    has_neg_force = True

    def force(self, rng, sc):
        a, b, c = force_size_chain(rng, sc)
        sc.color[c] = "red"
        sc.orientation[c] = rng.choice(TILTED)
        return a, b, c

    def neg_force(self, rng, sc, indices):
        c = indices[2]
        if rng.random() < 0.5:
            sc.orientation[c] = "upright"
        else:
            sc.color[c] = rng.choice(("blue", "green"))

    def plant_decoy(self, rng, sc, reserved):
        candidates = [i for i in range(sc.n_pieces) if i not in reserved]
        if rng.random() >= 0.85:
            return
        if len(candidates) >= 2:
            e, c = rng.sample(candidates, 2)
            sc.size[e] = rng.choice(MEDIUM_VALS)
            sc.size[c] = rng.choice(LARGE_VALS)
            sc.color[c] = "red"
            sc.orientation[c] = rng.choice(TILTED)
            sc.contacts.add(frozenset((e, c)))
            for f in sc.touches(e):  # no small root on the planted medium
                if f != c and f not in reserved and size_bucket(sc.size[f]) == "small":
                    sc.size[f] = rng.choice(MEDIUM_VALS)
        elif candidates:
            d = candidates[0]
            sc.size[d] = rng.choice(LARGE_VALS)
            sc.color[d] = "red"
            sc.orientation[d] = rng.choice(TILTED)

    def check(self, sc):
        return has_size_chain(sc, top=lambda c: sc.color[c] == "red" and sc.orientation[c] != "upright")


class MonoChainMinter(ZendoMinter):
    """zendo_mint6: a small -> medium -> large contact chain whose three pieces share one colour.
    The base vocabulary has only unary colour predicates, so the equality needs one clause per
    colour.

    Near misses recolour the top piece; decoys plant a same-colour small/medium/large triple
    where only the small-medium pair touches.
    """

    _max_vars, _max_body = 8, 11
    has_neg_force = True

    def force(self, rng, sc):
        a, b, c = force_size_chain(rng, sc)
        sc.color[a] = sc.color[b] = sc.color[c] = rng.choice(COLORS)
        return a, b, c

    def neg_force(self, rng, sc, indices):
        a, _b, c = indices
        sc.color[c] = rng.choice([x for x in COLORS if x != sc.color[a]])

    def plant_decoy(self, rng, sc, reserved):
        candidates = [i for i in range(sc.n_pieces) if i not in reserved]
        if len(candidates) < 3 or rng.random() >= 0.85:
            return
        p, q, r = rng.sample(candidates, 3)
        col = rng.choice(COLORS)
        sc.size[p] = rng.choice(SMALL_VALS)
        sc.size[q] = rng.choice(MEDIUM_VALS)
        sc.size[r] = rng.choice(LARGE_VALS)
        sc.color[p] = sc.color[q] = sc.color[r] = col
        sc.contacts.add(frozenset((p, q)))
        sc.contacts.discard(frozenset((q, r)))
        sc.contacts.discard(frozenset((p, r)))

    def check(self, sc):
        return has_size_chain(sc, same_color=True)
