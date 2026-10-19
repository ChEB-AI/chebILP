"""Michalski-trains minted problems (Larson & Michalski 1977; Cropper, AAAI 2022).

Each example is one train: two to four cars, each with a length, wheel count, roof and up to
three loads; a load has a shape and a quantity label. ``eastbound(Train)`` holds iff the train
satisfies a hidden rule, a disjunction of clauses that each require some cars (and loads) with
given properties. Car and load distributions follow Cropper's trains datasets.

Unlike the fixed zendo rules, the hidden rule is sampled from ``rule_seed`` under a difficulty
level, so a level defines a family of problems:

    level  clause body sizes   Cropper analogue (clauses / literals incl. heads)
    1      5                   trains1 (1 / 6)
    2      5, 4                trains2 (2 / 11)
    3      5, 5, 4             trains3 (3 / 17)
    4      6, 6, 5, 5          trains4 (4 / 26)
    5      6, 6, 6, 5, 5       -       (5 / 33)

A sampled clause must cover a non-trivial share of random trains, every one of its literals must
matter, and every clause must cover trains the others miss.
"""
import random
from dataclasses import dataclass

from chebILP.ilp_problem import ILPProblem
from chebILP.minted_ilp_problems.base_minter import ILPProblemMinter

# (attributes, weight) per car type, weights being the counts in Cropper's trains BK.
CAR_TYPES = (
    (("short", "two_wheels", "roof_closed", "roof_flat"), 534),
    (("short", "two_wheels", "roof_open"), 429),
    (("long", "two_wheels", "roof_open"), 387),
    (("long", "three_wheels", "roof_open"), 382),
    (("short", "two_wheels", "roof_closed", "roof_arc"), 333),
    (("short", "two_wheels", "roof_closed", "roof_peaked"), 206),
    (("long", "three_wheels", "roof_closed", "roof_jagged"), 204),
    (("long", "three_wheels", "roof_closed", "roof_flat"), 184),
    (("long", "two_wheels", "roof_closed", "roof_jagged"), 176),
    (("long", "two_wheels", "roof_closed", "roof_flat"), 174),
)
CAR_PREDICATES = ("short", "long", "two_wheels", "three_wheels", "roof_open", "roof_closed",
                  "roof_flat", "roof_arc", "roof_peaked", "roof_jagged")
SHAPES = (("circle", 1099), ("rectangle", 1090), ("hexagon", 608), ("diamond", 589),
          ("triangle", 584), ("inverted_triangle", 580))
QUANTITIES = (("one_load", 1745), ("two_load", 1657), ("three_load", 584), ("zero_load", 564))
LOADS_PER_CAR = ((0, 372), (1, 1104), (2, 1157), (3, 377))
LOAD_PREDICATES = tuple(s for s, _ in SHAPES) + tuple(q for q, _ in QUANTITIES)
MAX_LOADS = 3

DIFFICULTY_LEVELS = {
    1: (5,),
    2: (5, 4),
    3: (5, 5, 4),
    4: (6, 6, 5, 5),
    5: (6, 6, 6, 5, 5),
}


@dataclass
class Car:
    attrs: frozenset
    loads: list  # one frozenset of load predicates per load


@dataclass(frozen=True)
class CarSpec:
    """One ``has_car`` variable of a clause: required car attributes, and one frozenset of
    required load predicates per ``has_load`` variable."""
    attrs: frozenset
    loads: tuple

    def size(self) -> int:
        return 1 + len(self.attrs) + sum(1 + len(load) for load in self.loads)

    def matches(self, car: Car) -> bool:
        return self.attrs <= car.attrs and all(any(ls <= l for l in car.loads) for ls in self.loads)


# A clause is a tuple of CarSpecs. Variables are only shared through the train, so a clause holds
# iff every spec matches some car (two specs may match the same car, as under Prolog semantics).
def clause_holds(clause: tuple, train: list) -> bool:
    return all(any(spec.matches(car) for car in train) for spec in clause)


def clause_size(clause: tuple) -> int:
    return sum(spec.size() for spec in clause)


def clause_weakenings(clause: tuple) -> list:
    """Every clause with one attribute literal dropped; a spec left without literals is dropped too."""
    out = []
    for i, spec in enumerate(clause):
        rest = clause[:i] + clause[i + 1:]
        for a in spec.attrs:
            weaker = CarSpec(spec.attrs - {a}, spec.loads)
            out.append(rest + (weaker,) if weaker.attrs or weaker.loads else rest)
        for j, load in enumerate(spec.loads):
            for a in load:
                loads = spec.loads[:j] + ((load - {a},) if len(load) > 1 else ()) + spec.loads[j + 1:]
                weaker = CarSpec(spec.attrs, loads)
                out.append(rest + (weaker,) if weaker.attrs or weaker.loads else rest)
    return out


def clause_to_prolog(clause: tuple, head: str) -> str:
    names = iter("BCDEFGHIJKLMNOPQRSTUVWXYZ")
    body = []
    for spec in clause:
        c = next(names)
        body.append(f"has_car(A,{c})")
        body += [f"{a}({c})" for a in sorted(spec.attrs)]
        for load in spec.loads:
            l = next(names)
            body.append(f"has_load({c},{l})")
            body += [f"{a}({l})" for a in sorted(load)]
    return f"{head}(A) :- {', '.join(body)}."


def _weighted(rng: random.Random, table):
    values, weights = zip(*table)
    return rng.choices(values, weights=weights)[0]


def random_load(rng: random.Random) -> frozenset:
    return frozenset((_weighted(rng, SHAPES), _weighted(rng, QUANTITIES)))


def random_car(rng: random.Random) -> Car:
    return Car(frozenset(_weighted(rng, CAR_TYPES)),
               [random_load(rng) for _ in range(_weighted(rng, LOADS_PER_CAR))])


def random_train(rng: random.Random, n_cars: tuple) -> list:
    return [random_car(rng) for _ in range(rng.randint(*n_cars))]


def car_matching(rng: random.Random, spec: CarSpec, max_tries: int = 200) -> Car | None:
    """A random car conditioned on matching ``spec``; None if no car type or load can."""
    types = [(attrs, w) for attrs, w in CAR_TYPES if spec.attrs <= set(attrs)]
    if not types or len(spec.loads) > MAX_LOADS:
        return None
    car = Car(frozenset(_weighted(rng, types)), [random_load(rng) for _ in range(_weighted(rng, LOADS_PER_CAR))])
    free = list(range(len(car.loads)))
    for ls in spec.loads:
        load = next((l for l in (random_load(rng) for _ in range(max_tries)) if ls <= l), None)
        if load is None:
            return None
        if len(car.loads) < MAX_LOADS:
            car.loads.append(load)
        else:
            car.loads[free.pop(rng.randrange(len(free)))] = load
    return car


def force_clause(rng: random.Random, train: list, clause: tuple, reserved: set) -> set:
    """Overwrites distinct unreserved cars so each spec of ``clause`` matches one; returns the
    cars it used (none if the train is too short or a spec is unsatisfiable)."""
    free = [i for i in range(len(train)) if i not in reserved]
    if len(free) < len(clause):
        return set()
    cars = [car_matching(rng, spec) for spec in clause]
    if any(car is None for car in cars):
        return set()
    used = rng.sample(free, len(clause))
    for i, car in zip(used, cars):
        train[i] = car
    return set(used)


class TrainsMinter(ILPProblemMinter):
    """Samples a hidden rule from ``rule_seed`` (or takes ``rule``, a list of clauses) and mints
    trains for it.

    Args:
        level: a key of ``DIFFICULTY_LEVELS``; ignored if ``body_sizes`` or ``rule`` is given.
        body_sizes: number of body literals per clause, one entry per clause.
        rule_seed: seeds the rule; ``mint``'s seed only resamples the trains.
        n_cars: (min, max) cars per train.
        max_cars_per_clause: ``has_car`` literals a sampled clause may use.
        min_coverage / max_coverage: share of random trains a sampled clause may cover.
        max_rule_coverage: share of random trains the whole rule may cover.
        force_prob: chance a train has one clause planted (otherwise positives are rare).
        near_miss_prob: chance a planted clause is then broken by resampling one of its cars.
        decoy_prob: chance a train gets a clause with one literal dropped planted elsewhere,
            so that partial patterns are not discriminative.
    """

    problem_type = "trains"
    target = "eastbound"

    def __init__(self, out_root=None, level: int = 1, body_sizes: tuple | None = None, rule_seed: int = 0,
                 rule: list | None = None, n_cars: tuple = (2, 4), max_cars_per_clause: int = 3,
                 min_coverage: float = 0.03, max_coverage: float = 0.35, max_rule_coverage: float = 0.8,
                 force_prob: float = 0.5, near_miss_prob: float = 0.3, decoy_prob: float = 0.5,
                 n_reference: int = 3000, library_root=None):
        super().__init__(out_root, library_root)
        self.n_cars = n_cars
        self.force_prob = force_prob
        self.near_miss_prob = near_miss_prob
        self.decoy_prob = decoy_prob
        if rule is None:
            body_sizes = tuple(body_sizes or DIFFICULTY_LEVELS[level])
            rng = random.Random(f"{rule_seed}-{body_sizes}")
            reference = [random_train(rng, n_cars) for _ in range(n_reference)]
            rule = self.sample_rule(rng, body_sizes, reference, max_cars_per_clause,
                                    min_coverage, max_coverage, max_rule_coverage)
        self.rule = [tuple(clause) for clause in rule]

    @staticmethod
    def random_clause(rng: random.Random, max_cars: int) -> tuple:
        specs = []
        for _ in range(rng.randint(1, max_cars)):
            attrs = frozenset(rng.sample(CAR_PREDICATES, rng.choice((0, 1, 1, 2, 2, 3))))
            loads = tuple(frozenset(rng.sample(LOAD_PREDICATES, rng.choice((1, 1, 2))))
                          for _ in range(rng.choice((0, 0, 1, 1, 2))))
            if attrs or loads:
                specs.append(CarSpec(attrs, loads))
        return tuple(specs)

    @classmethod
    def sample_rule(cls, rng, body_sizes, reference, max_cars, min_cov, max_cov, max_rule_cov,
                    max_tries: int = 200_000) -> list:
        n = len(reference)

        def covered(clause):
            return {k for k, train in enumerate(reference) if clause_holds(clause, train)}

        rule, covers = [], []
        tries = 0
        for size in body_sizes:
            while True:
                tries += 1
                if tries > max_tries:
                    raise RuntimeError(f"no rule with body sizes {body_sizes} after {max_tries} tries")
                clause = cls.random_clause(rng, max_cars)
                if clause_size(clause) != size:
                    continue
                cov = covered(clause)
                if not min_cov * n <= len(cov) <= max_cov * n:
                    continue
                # every literal must matter, which also rules out contradictory specs
                if any(len(covered(w)) <= len(cov) for w in clause_weakenings(clause)):
                    continue
                others = set().union(*covers)
                if len(cov - others) < min_cov * n / 2 or any(len(c - cov) < min_cov * n / 2 for c in covers):
                    continue
                if len(others | cov) > max_rule_cov * n:
                    continue
                rule.append(clause)
                covers.append(cov)
                break
        return rule

    def check(self, train: list) -> bool:
        return any(clause_holds(clause, train) for clause in self.rule)

    def rule_str(self) -> str:
        return "\n".join(clause_to_prolog(clause, self.target) for clause in self.rule)

    def max_vars(self):
        return 1 + max(sum(1 + len(spec.loads) for spec in clause) for clause in self.rule)

    def max_body(self):
        return max(clause_size(clause) for clause in self.rule)

    def sample_train(self, rng: random.Random) -> list:
        train = random_train(rng, self.n_cars)
        reserved = set()
        if rng.random() < self.force_prob:
            clause = rng.choice(self.rule)
            reserved = force_clause(rng, train, clause, set())
            if reserved and rng.random() < self.near_miss_prob:
                i = rng.choice(sorted(reserved))
                train[i] = random_car(rng)
        if rng.random() < self.decoy_prob:
            weakenings = clause_weakenings(rng.choice(self.rule))
            if weakenings:
                force_clause(rng, train, rng.choice(weakenings), reserved)
        return train

    def mint(self, seed, n_pos: int = 500, n_neg: int = 500, max_attempts: int = 1_000_000) -> ILPProblem:
        """Trains 0..n_pos-1 are positive, the rest negative; car j of train t is ``t<t>_c<j>`` and
        its load k is ``t<t>_c<j>_l<k>``."""
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
            raise RuntimeError(f"{type(self).__name__}: only got {len(pos)} pos / {len(neg)} neg "
                               f"after {attempts} attempts")

        bk_facts = {p: [] for p in ("has_car", "has_load") + CAR_PREDICATES + LOAD_PREDICATES}
        for t, train in enumerate(pos + neg):
            for j, car in enumerate(train):
                c = f"t{t}_c{j}"
                bk_facts["has_car"].append((f"t{t}", c))
                for a in car.attrs:
                    bk_facts[a].append(c)
                for k, load in enumerate(car.loads):
                    l = f"{c}_l{k}"
                    bk_facts["has_load"].append((c, l))
                    for a in load:
                        bk_facts[a].append(l)
        trains = [f"t{t}" for t in range(n_pos + n_neg)]
        return ILPProblem((self.target, 1), trains[:n_pos], trains[n_pos:], bk_facts)

    def get_task_description(self):
        return f"""\
Michalski's trains: each example is a train, and a hidden rule decides whether it is eastbound.
A train has several cars (has_car/2). Every car is short or long, has two or three wheels, and
has an open or a closed roof; a closed roof is additionally flat, arc-shaped, peaked or jagged.
A car carries up to three loads (has_load/2); every load has a shape and a quantity label.

The hidden rule for {self.target}(Train) is a disjunction of one or more clauses. Each clause
asks for one or more cars of the train (and possibly loads on them) with certain properties.
Recover the clauses that separate the eastbound from the other trains, and express them as
rules for {self.target}(Train), using at most one train variable per rule."""

    def get_glossary(self):
        return """\
eastbound(Train)          [TARGET] the train satisfies the hidden rule.
has_car(Train, Car)       Car is a car of Train.
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
