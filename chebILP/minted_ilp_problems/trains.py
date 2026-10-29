"""Michalski-trains minted problems (Larson & Michalski 1977; Cropper, AAAI 2022).

Each example is a train of three to seven cars linked by ``infront/2``; a car has a length, wheel
count, roof and up to three loads, a load a shape and a quantity label. ``eastbound(Train)`` holds
iff the train satisfies a hidden rule.

``ChainedTrainsMinter`` samples the rule per ``CHAIN_LEVELS``: clauses over runs of consecutive
cars, so they have to be learned as chains through ``infront``. ``AheadMinter`` and
``ShapeRunMinter`` are fixed rules that need a disjunction of long clauses in the base vocabulary
but are short with an invented predicate.
"""

import random
from dataclasses import dataclass

from chebILP.ilp_problem import ILPProblem
from chebILP.minted_ilp_problems.base_minter import ILPProblemMinter

LENGTHS = ("short", "long")
WHEELS = ("two_wheels", "three_wheels")
ROOFS = ("roof_open", "roof_flat", "roof_arc", "roof_peaked", "roof_jagged")
CAR_PREDICATES = LENGTHS + WHEELS + ROOFS + ("roof_closed",)
SHAPES = ("circle", "rectangle", "triangle", "inverted_triangle", "diamond", "hexagon")
QUANTITIES = ("zero_load", "one_load", "two_load", "three_load")
LOAD_PREDICATES = SHAPES + QUANTITIES
N_CARS = (3, 7)
MAX_LOADS = 3

# One (chain lengths, body size) per clause: (3,) is a run of three consecutive cars, (2, 1) a run
# of two plus one car anywhere.
CHAIN_LEVELS = {
    1: (((3,), 7),),
    2: (((3,), 9),),
    3: (((4,), 10),),
    4: (((3,), 9), ((2, 1), 8)),
    5: (((4,), 10), ((3,), 9), ((2, 2), 9)),
}

# Share of random trains a clause / the whole rule may cover.
MIN_COVERAGE, MAX_COVERAGE, MAX_RULE_COVERAGE = 0.03, 0.35, 0.8
N_REFERENCE = 3000
# Chance a clause is planted, then broken again (near miss), and that a weakened clause is planted
# as a decoy; a decoy is the clause with one infront link cut with SPLIT_DECOY_PROB.
FORCE_PROB, NEAR_MISS_PROB, DECOY_PROB, SPLIT_DECOY_PROB = 0.5, 0.3, 0.5, 0.5
PLACEMENT_TRIES = 50


@dataclass
class Car:
    attrs: frozenset
    loads: list  # one frozenset of load predicates per load


@dataclass(frozen=True)
class CarSpec:
    """Required car attributes, and one frozenset of required load predicates per load. An empty
    spec matches any car."""

    attrs: frozenset
    loads: tuple

    def size(self) -> int:
        # counts the has_car or infront literal binding the car
        return 1 + len(self.attrs) + sum(1 + len(load) for load in self.loads)

    def is_empty(self) -> bool:
        return not self.attrs and not self.loads

    def matches(self, car: Car) -> bool:
        return self.attrs <= car.attrs and all(
            any(ls <= ld for ld in car.loads) for ls in self.loads
        )


# A clause is a tuple of chains, a chain a tuple of CarSpecs matching consecutive cars. Chains are
# only linked through the train, so they may use the same cars, as under Prolog semantics.
def chain_holds(chain: tuple, train: list) -> bool:
    return any(
        all(spec.matches(train[i + j]) for j, spec in enumerate(chain))
        for i in range(len(train) - len(chain) + 1)
    )


def clause_holds(clause: tuple, train: list) -> bool:
    return all(chain_holds(chain, train) for chain in clause)


def clause_size(clause: tuple) -> int:
    return sum(spec.size() for chain in clause for spec in chain)


def clause_vars(clause: tuple) -> int:
    return 1 + sum(1 + len(spec.loads) for chain in clause for spec in chain)


def spec_weakenings(spec: CarSpec) -> list:
    out = [CarSpec(spec.attrs - {a}, spec.loads) for a in sorted(spec.attrs)]
    for j, load in enumerate(spec.loads):
        for a in sorted(load):
            loads = (
                spec.loads[:j]
                + ((load - {a},) if len(load) > 1 else ())
                + spec.loads[j + 1 :]
            )
            out.append(CarSpec(spec.attrs, loads))
    return out


def trim_chain(chain: tuple) -> tuple:
    start, end = 0, len(chain)
    while start < end and chain[start].is_empty():
        start += 1
    while end > start and chain[end - 1].is_empty():
        end -= 1
    return chain[start:end]


def chain_splits(clause: tuple) -> list:
    """Every clause with one infront link cut."""
    out = []
    for i, chain in enumerate(clause):
        rest = clause[:i] + clause[i + 1 :]
        for j in range(1, len(chain)):
            out.append(
                rest
                + tuple(h for h in (trim_chain(chain[:j]), trim_chain(chain[j:])) if h)
            )
    return out


def clause_weakenings(clause: tuple) -> list:
    """Every clause with one literal dropped."""
    out = []
    for i, chain in enumerate(clause):
        rest = clause[:i] + clause[i + 1 :]
        for j, spec in enumerate(chain):
            for weaker in spec_weakenings(spec):
                trimmed = trim_chain(chain[:j] + (weaker,) + chain[j + 1 :])
                out.append(rest + ((trimmed,) if trimmed else ()))
    return out + chain_splits(clause)


def clause_to_prolog(clause: tuple, head: str) -> str:
    names = iter("BCDEFGHIJKLMNOPQRSTUVWXYZ")
    body = []
    for chain in clause:
        prev = None
        for spec in chain:
            c = next(names)
            body.append(f"has_car(A,{c})" if prev is None else f"infront({prev},{c})")
            body += [f"{a}({c})" for a in sorted(spec.attrs)]
            for load in spec.loads:
                lid = next(names)
                body.append(f"has_load({c},{lid})")
                body += [f"{a}({lid})" for a in sorted(load)]
            prev = c
    return f"{head}(A) :- {', '.join(body)}."


def random_attrs(rng: random.Random) -> frozenset:
    roof = rng.choice(ROOFS)
    closed = ("roof_closed",) if roof != "roof_open" else ()
    return frozenset((rng.choice(LENGTHS), rng.choice(WHEELS), roof) + closed)


def random_load(rng: random.Random) -> frozenset:
    return frozenset((rng.choice(SHAPES), rng.choice(QUANTITIES)))


def random_car(rng: random.Random) -> Car:
    return Car(
        random_attrs(rng), [random_load(rng) for _ in range(rng.randint(0, MAX_LOADS))]
    )


def random_train(rng: random.Random) -> list:
    return [random_car(rng) for _ in range(rng.randint(*N_CARS))]


def car_matching(rng: random.Random, spec: CarSpec, max_tries: int = 200) -> Car | None:
    """A random car conditioned on matching ``spec``; None if no car or load can."""
    if len(spec.loads) > MAX_LOADS:
        return None
    attrs = next(
        (a for a in (random_attrs(rng) for _ in range(max_tries)) if spec.attrs <= a),
        None,
    )
    loads = [
        next(
            (ld for ld in (random_load(rng) for _ in range(max_tries)) if ls <= ld),
            None,
        )
        for ls in spec.loads
    ]
    if attrs is None or None in loads:
        return None
    loads += [random_load(rng) for _ in range(rng.randint(0, MAX_LOADS - len(loads)))]
    rng.shuffle(loads)
    return Car(attrs, loads)


def force_clause(rng: random.Random, train: list, clause: tuple, reserved: set) -> set:
    """Overwrites unreserved cars so every chain of ``clause`` matches a run of its own; returns the
    cars it used (none if it could not)."""
    free = [i for i in range(len(train)) if i not in reserved]
    if len(free) < sum(map(len, clause)):
        return set()
    cars = [car_matching(rng, spec) for chain in clause for spec in chain]
    if any(car is None for car in cars):
        return set()
    free_set = set(free)
    for _ in range(PLACEMENT_TRIES):
        starts = rng.sample(free, len(clause))
        cells = [s + j for s, chain in zip(starts, clause) for j in range(len(chain))]
        if len(set(cells)) == len(cells) and free_set.issuperset(cells):
            for i, car in zip(cells, cars):
                train[i] = car
            return set(cells)
    return set()


def random_spec(rng: random.Random) -> CarSpec:
    attrs = frozenset(rng.sample(CAR_PREDICATES, rng.choice((0, 1, 1, 2))))
    loads = tuple(
        frozenset(rng.sample(LOAD_PREDICATES, rng.choice((1, 1, 2))))
        for _ in range(rng.choice((0, 0, 1)))
    )
    return CarSpec(attrs, loads)


def random_clause(rng: random.Random, chain_lengths: tuple) -> tuple:
    """One chain per length; empty (so rejected) if a chain's end car carries no literal."""
    clause = tuple(tuple(random_spec(rng) for _ in range(k)) for k in chain_lengths)
    return clause if all(trim_chain(chain) == chain for chain in clause) else ()


def sample_rule(
    rng: random.Random, clause_shapes: tuple, max_tries: int = 500_000
) -> list:
    """One clause per ``(chain_lengths, body_size)``. A clause must cover a non-trivial share of
    random trains, every literal and infront link must matter, and every clause must cover trains
    the others miss."""
    reference = [random_train(rng) for _ in range(N_REFERENCE)]
    n = len(reference)

    def covered(clause):
        return {k for k, train in enumerate(reference) if clause_holds(clause, train)}

    rule, covers = [], []
    tries = 0
    for chain_lengths, size in clause_shapes:
        while True:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(
                    f"no rule with clause shapes {clause_shapes} after {max_tries} tries"
                )
            clause = random_clause(rng, chain_lengths)
            if clause_size(clause) != size:
                continue
            cov = covered(clause)
            if not MIN_COVERAGE * n <= len(cov) <= MAX_COVERAGE * n:
                continue
            if any(len(covered(w)) <= len(cov) for w in clause_weakenings(clause)):
                continue
            others = set().union(*covers)
            min_new = MIN_COVERAGE * n / 2
            if len(cov - others) < min_new or any(
                len(c - cov) < min_new for c in covers
            ):
                continue
            if len(others | cov) > MAX_RULE_COVERAGE * n:
                continue
            rule.append(clause)
            covers.append(cov)
            break
    return rule


class TrainsMinter(ILPProblemMinter):
    """Shared BK, vocabulary and rejection sampling; a subclass defines ``check`` (the label) and
    ``sample_train``."""

    problem_type = "trains"
    target = "eastbound"

    def check(self, train: list) -> bool:
        raise NotImplementedError

    def sample_train(self, rng: random.Random) -> list:
        raise NotImplementedError

    def mint(
        self, seed, n_pos: int = 500, n_neg: int = 500, max_attempts: int = 1_000_000
    ) -> ILPProblem:
        """Trains 0..n_pos-1 are positive, the rest negative; car j of train t is ``t<t>_c<j>``,
        its load k ``t<t>_c<j>_l<k>``."""
        rng = random.Random(seed)
        pos, neg = [], []
        attempts = 0
        while (len(pos) < n_pos or len(neg) < n_neg) and attempts < max_attempts:
            attempts += 1
            train = self.sample_train(rng)
            label = self.check(train)
            if label and len(pos) < n_pos:
                pos.append(train)
            elif not label and len(neg) < n_neg:
                neg.append(train)
        if len(pos) < n_pos or len(neg) < n_neg:
            raise RuntimeError(
                f"{type(self).__name__}: only got {len(pos)} pos / {len(neg)} neg "
                f"after {attempts} attempts"
            )

        bk_facts = {
            p: []
            for p in ("has_car", "infront", "has_load")
            + CAR_PREDICATES
            + LOAD_PREDICATES
        }
        for t, train in enumerate(pos + neg):
            for j, car in enumerate(train):
                c = f"t{t}_c{j}"
                bk_facts["has_car"].append((f"t{t}", c))
                if j + 1 < len(train):
                    bk_facts["infront"].append((c, f"t{t}_c{j + 1}"))
                for a in sorted(car.attrs):
                    bk_facts[a].append(c)
                for k, load in enumerate(car.loads):
                    lid = f"{c}_l{k}"
                    bk_facts["has_load"].append((c, lid))
                    for a in sorted(load):
                        bk_facts[a].append(lid)
        trains = [f"t{t}" for t in range(n_pos + n_neg)]
        return ILPProblem((self.target, 1), trains[:n_pos], trains[n_pos:], bk_facts)

    def get_task_description(self):
        return f"""\
Michalski's trains: each example is a train, and a hidden rule decides whether it is eastbound.
A train has three to seven cars in a fixed order (has_car/2); infront/2 links each car to the car
directly behind it. Every car is short or long, has two or three wheels, and has an open or a
closed roof; a closed roof is additionally flat, arc-shaped, peaked or jagged.
A car carries up to three loads (has_load/2); every load has a shape and a quantity label.

The hidden rule for {self.target}(Train) is a disjunction of one or more clauses. Each clause asks
for one or more cars of the train (and possibly loads on them) with certain properties, possibly
in a certain order.
Recover the clauses that separate the eastbound from the other trains, and express them as
rules for {self.target}(Train), using at most one train variable per rule."""

    def get_glossary(self):
        return """\
eastbound(Train)          [TARGET] the train satisfies the hidden rule.
has_car(Train, Car)       Car is a car of Train.
infront(Car1, Car2)       Car1 is directly in front of Car2 (same train).
has_load(Car, Load)       Load is carried by Car.
short(Car) / long(Car)    the length of Car.
two_wheels(Car) / three_wheels(Car)
                          the number of wheels of Car.
roof_open(Car) / roof_closed(Car)
                          Car has an open / a closed roof.
roof_flat(Car) / roof_arc(Car) / roof_peaked(Car) / roof_jagged(Car)
                          the shape of a closed roof.
circle(Load) / rectangle(Load) / triangle(Load) / inverted_triangle(Load) / diamond(Load) / hexagon(Load)
                          the shape of Load.
zero_load(Load) / one_load(Load) / two_load(Load) / three_load(Load)
                          the quantity label of Load."""


class ChainedTrainsMinter(TrainsMinter):
    """trains_mint1a-1e: samples a rule per ``CHAIN_LEVELS[level]``; ``rule_seed`` seeds the rule,
    ``mint``'s seed only the trains. Decoys plant the right cars out of order."""

    def __init__(
        self, out_root=None, level: int = 1, rule_seed: int = 0, library_root=None
    ):
        super().__init__(out_root, library_root)
        self.rule = sample_rule(
            random.Random(f"{rule_seed}-{('chain',) + CHAIN_LEVELS[level]}"),
            CHAIN_LEVELS[level],
        )

    def check(self, train):
        return any(clause_holds(clause, train) for clause in self.rule)

    def rule_str(self):
        return "\n".join(clause_to_prolog(clause, self.target) for clause in self.rule)

    def max_vars(self):
        return max(clause_vars(clause) for clause in self.rule)

    def max_body(self):
        return max(clause_size(clause) for clause in self.rule)

    def sample_train(self, rng):
        train = random_train(rng)
        reserved = set()
        if rng.random() < FORCE_PROB:
            reserved = force_clause(rng, train, rng.choice(self.rule), set())
            if reserved and rng.random() < NEAR_MISS_PROB:
                train[rng.choice(sorted(reserved))] = random_car(rng)
        if rng.random() < DECOY_PROB:
            clause = rng.choice(self.rule)
            splits = chain_splits(clause)
            weakenings = (
                splits
                if splits and rng.random() < SPLIT_DECOY_PROB
                else clause_weakenings(clause)
            )
            if weakenings:
                force_clause(rng, train, rng.choice(weakenings), reserved)
        return train


class AheadMinter(TrainsMinter):
    """trains_mint2: a short car with a jagged roof is somewhere ahead of a car carrying a hexagon load.

        eastbound(A) :- has_car(A,B), short(B), roof_jagged(B), ahead(B,C), has_load(C,D), hexagon(D).
        ahead(B,C) :- infront(B,C).
        ahead(B,C) :- infront(B,D), ahead(D,C).

    Without ``ahead`` this needs one clause per distance 1..6. Planted distances are uniform; near
    misses plant the two cars in the opposite order.
    """

    FRONT = CarSpec(frozenset({"short", "roof_jagged"}), ())
    REAR = CarSpec(frozenset(), (frozenset({"hexagon"}),))

    def check(self, train):
        fronts = [i for i, car in enumerate(train) if self.FRONT.matches(car)]
        return bool(fronts) and any(
            self.REAR.matches(car) for car in train[fronts[0] + 1 :]
        )

    def sample_train(self, rng):
        train = random_train(rng)
        if rng.random() < FORCE_PROB:
            d = rng.randint(1, len(train) - 1)
            i = rng.randint(0, len(train) - 1 - d)
            front, rear = (i, i + d) if rng.random() >= NEAR_MISS_PROB else (i + d, i)
            train[front] = car_matching(rng, self.FRONT)
            train[rear] = car_matching(rng, self.REAR)
        return train

    def rule_str(self):
        return (
            "eastbound(A) :- has_car(A,B), short(B), roof_jagged(B), ahead(B,C), has_load(C,D), hexagon(D).\n"
            "ahead(B,C) :- infront(B,C).\n"
            "ahead(B,C) :- infront(B,D), ahead(D,C)."
        )

    def max_vars(self):
        return 2 + N_CARS[1]

    def max_body(self):
        return 4 + N_CARS[1]


def load_shapes(car: Car) -> set:
    return {a for load in car.loads for a in load if a in SHAPES}


class ShapeRunMinter(TrainsMinter):
    """trains_mint3: three consecutive cars each carry a load of one common shape.

        eastbound(A) :- has_car(A,B), infront(B,C), infront(C,D), has_load(B,E), has_load(C,F),
                        has_load(D,G), same_shape(E,F), same_shape(F,G).

    Without ``same_shape`` this needs one clause per shape. Near misses leave a gap before the
    third car; decoys plant two-car runs.
    """

    RUN = 3

    def check(self, train):
        return any(
            set.intersection(*(load_shapes(car) for car in train[i : i + self.RUN]))
            for i in range(len(train) - self.RUN + 1)
        )

    def _plant(self, rng, train, cells, shape):
        for i in cells:
            train[i] = car_matching(rng, CarSpec(frozenset(), (frozenset({shape}),)))

    def sample_train(self, rng):
        train = random_train(rng)
        n = len(train)
        if rng.random() < FORCE_PROB:
            if rng.random() < NEAR_MISS_PROB and n > self.RUN:
                i = rng.randint(0, n - self.RUN - 1)
                cells = list(range(i, i + self.RUN - 1)) + [i + self.RUN]
            else:
                i = rng.randint(0, n - self.RUN)
                cells = list(range(i, i + self.RUN))
            self._plant(rng, train, cells, rng.choice(SHAPES))
        if rng.random() < DECOY_PROB:
            i = rng.randint(0, n - 2)
            self._plant(rng, train, [i, i + 1], rng.choice(SHAPES))
        return train

    def rule_str(self):
        return "\n".join(
            f"eastbound(A) :- has_car(A,B), infront(B,C), infront(C,D), has_load(B,E), {s}(E), "
            f"has_load(C,F), {s}(F), has_load(D,G), {s}(G)."
            for s in SHAPES
        )

    def max_vars(self):
        return 7

    def max_body(self):
        return 9
