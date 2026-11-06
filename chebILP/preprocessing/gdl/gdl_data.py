"""The IGGP games as a dataset: one class per target of a game, ``gdl_<game>_<target>``.

``molecules`` holds one row per sampled state: ``<game>_s<i>`` for a state, ``<game>_m<i>`` for a
(state, joint move) pair. The ``next`` targets use the move rows, the others the state rows. A row's
``mol`` holds its ``true_*`` (and ``does_*``) facts without the state argument.

An example is an atom of the target, ``gdl_tictactoe_legal_mark(tictactoe_s3,xplayer,1,1)``, with
id ``tictactoe_s3,xplayer,1,1``. A state contributes every candidate atom of the target (see
:meth:`GameTarget.candidates`), positive iff the game derives it. Negatives are never enumerated:
:meth:`GDLDataset.split_examples` samples them by index.
"""

import bisect
import itertools
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import pandas as pd
import tqdm

from chebILP.preprocessing.base_dataset import BaseDataset, PromptSpec
from chebILP.preprocessing.gdl import llm_prompt
from chebILP.preprocessing.gdl.gdl import flat_atom, flat_value, target_atoms
from chebILP.preprocessing.gdl.games import (
    IGGP_ROOT,
    GameTarget,
    gdl_targets,
    iggp_games,
    load_game,
    sample_steps,
    typing_predicates,
)

GDL_ROOT = "gdl"
SAMPLE_SEED = 20260907
N_STATES = 500
MAX_STEPS = 200
MIN_POS_PER_SPLIT = 3
EVAL_CAP = 1000


def game_name(game: str) -> str:
    """A game's name in row and class ids."""
    return re.sub(r"\W", "_", game.lower())


def _fact(pred: str, args: tuple) -> str:
    return f"{pred}({','.join(str(a) for a in args)})."


def _example_id(row: str, args: tuple) -> str:
    return ",".join([row, *(str(a) for a in args)])


def _build_game(game: str, root: str) -> tuple[list[dict], dict, dict]:
    """A game's rows, static facts and targets (see :meth:`GDLDataset._load_or_build`)."""
    g = load_game(game, root)
    name = game_name(game)
    statics = {
        pred: [a if isinstance(a, tuple) else (a,) for a in facts]
        for pred, facts in {
            **g.static_relations(),
            **typing_predicates(game, root),
        }.items()
    }
    rows, steps = [], {}
    for kind, with_moves in (("s", False), ("m", True)):
        steps[kind] = []
        for i, (state, move, info) in enumerate(
            sample_steps(game, root, SAMPLE_SEED, with_moves, N_STATES, MAX_STEPS)
        ):
            facts = [flat_atom("true", f) for f in sorted(state, key=str)]
            facts += [
                flat_atom("does", m, (flat_value(r),))
                for r, m in sorted((move or {}).items(), key=str)
            ]
            row = f"{name}_{kind}{i}"
            rows.append({"id": row, "game": game, "kind": kind, "mol": facts})
            steps[kind].append((row, info))

    targets = {}
    for target in gdl_targets(game, root):
        t = GameTarget(game, target, root)
        kind = "m" if t.with_moves else "s"
        holds = [target_atoms(info, target) for _, info in steps[kind]]
        candidates = t.candidates(holds)
        index = {c: i for i, c in enumerate(candidates)}
        class_id = f"gdl_{name}_{target}"
        targets[class_id] = {
            "game": game,
            "target": target,
            "kind": kind,
            "arity": 1 + (len(candidates[0]) if candidates else 0),
            "candidates": candidates,
            "positives": {
                row: sorted(index[c] for c in h)
                for (row, _), h in zip(steps[kind], holds)
                if h
            },
            "definition": t.description(class_id),
            "glossary": t.glossary(class_id),
            "head_args": t.head_args(),
        }
    return rows, statics, targets


class GDLDataset(BaseDataset):
    """GDL game targets; see the module docstring."""

    DATASET_NAME: str = "gdl"
    eval_cap = EVAL_CAP

    def __init__(
        self,
        version: str = "1",
        base_dir: str = "data",
        min_pos_samples: int = 25,
        iggp_root: str = IGGP_ROOT,
        workers: int = 8,
    ):
        super().__init__(
            version=version, base_dir=base_dir, min_pos_samples=min_pos_samples
        )
        self.iggp_root = iggp_root
        self.workers = workers
        self.games_path = os.path.join(self.data_dir, "games.pkl")

    def download(self) -> None:
        if not iggp_games(self.iggp_root):
            raise FileNotFoundError(
                f"No IGGP games under {self.iggp_root}; run "
                f"`git clone --depth 1 https://github.com/andrewcropper/iggp {self.iggp_root}`."
            )

    def process_data(self) -> None:
        data = self._load_or_build()
        self.molecules = data["rows"]
        self.statics = data["statics"]
        self.targets = data["targets"]
        self._row_position = {r: i for i, r in enumerate(self.molecules.index)}
        self._rows_by_kind = {
            key: list(group.index)
            for key, group in self.molecules.groupby(["game", "kind"], sort=False)
        }
        self.graph = self._build_graph()

    def _load_or_build(self) -> dict:
        """``rows`` (the row frame), ``statics`` (game -> predicate -> argument tuples) and
        ``targets`` (class id -> game, target, row kind, arity, candidate atoms, positive candidate
        indices per row, prompt text), cached in ``games.pkl``."""
        if os.path.exists(self.games_path):
            print(f"Loading games from {self.games_path}...")
            return pd.read_pickle(self.games_path)
        games = iggp_games(self.iggp_root)
        rows, statics, targets = [], {}, {}
        with ProcessPoolExecutor(self.workers) as pool:
            results = pool.map(_build_game, games, itertools.repeat(self.iggp_root))
            for game, (r, s, t) in tqdm.tqdm(
                zip(games, results), total=len(games), desc="Simulating games"
            ):
                rows += r
                statics[game] = s
                targets.update(t)
        frame = pd.DataFrame(rows).set_index("id")
        frame.index.name = None
        assert frame.index.is_unique, "row ids collide after sanitising game names"
        data = {"rows": frame, "statics": statics, "targets": targets}
        os.makedirs(self.data_dir, exist_ok=True)
        pd.to_pickle(data, self.games_path)
        print(f"{len(frame)} rows, {len(targets)} targets of {len(games)} games")
        return data

    def _build_graph(self) -> nx.DiGraph:
        """Target ``gdl_<game>_<target>`` -> game ``gdl_<game>`` -> root ``gdl``."""
        graph = nx.DiGraph()
        graph.add_node(GDL_ROOT, name="GDL game", definition=None)
        for class_id, t in self.targets.items():
            game = f"gdl_{game_name(t['game'])}"
            if game not in graph:
                graph.add_node(game, name=t["game"], definition=None)
                graph.add_edge(game, GDL_ROOT)
            graph.add_node(
                class_id,
                name=f"{t['game']}: {t['target']}",
                definition=t["definition"],
            )
            graph.add_edge(class_id, game)
        return graph

    def _rows(self, class_id: str) -> list[str]:
        t = self.targets[class_id]
        return self._rows_by_kind.get((t["game"], t["kind"]), [])

    # --- labels and splits -------------------------------------------------------

    def select_labels(self) -> list[str]:
        """Targets with at least ``min_pos_samples`` positive atoms."""
        labels = [
            c
            for c, t in self.targets.items()
            if sum(map(len, t["positives"].values())) >= self.min_pos_samples
        ]
        labels = self.sort_labels(labels)
        self._write_labels(labels)
        return labels

    def _write_labels(self, labels: list[str]) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(self.labels_file, "w") as f:
            f.write("\n".join(labels) + "\n")
        print(f"Saved {len(labels)} labels to {self.labels_file}")

    def create_splits(
        self,
        labels: list[str],
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Split each game's rows of each kind at random, so all atoms of a state share a split.

        Labels with fewer than ``MIN_POS_PER_SPLIT`` positives in some split are dropped from
        ``labels.txt``.
        """
        rng = random.Random(seed)
        split_of = {}
        for rows in self._rows_by_kind.values():
            rows = list(rows)
            rng.shuffle(rows)
            n_val, n_test = round(len(rows) * val_ratio), round(len(rows) * test_ratio)
            for i, row in enumerate(rows):
                split_of[row] = (
                    "validation"
                    if i < n_val
                    else "test"
                    if i < n_val + n_test
                    else "train"
                )

        def testable(class_id):
            counts = dict.fromkeys(("train", "validation", "test"), 0)
            for row, idx in self.targets[class_id]["positives"].items():
                counts[split_of[row]] += len(idx)
            return min(counts.values()) >= MIN_POS_PER_SPLIT

        dropped = [c for c in labels if not testable(c)]
        if dropped:
            print(
                f"Dropped {len(dropped)} targets with fewer than {MIN_POS_PER_SPLIT} positives "
                f"in some split: {', '.join(dropped)}"
            )
        self._write_labels([c for c in labels if c not in dropped])

        splits = pd.DataFrame(
            [(r, split_of[r]) for r in self.molecules.index], columns=["id", "split"]
        )
        splits.to_csv(self.splits_file, index=False)
        n = splits["split"].value_counts()
        print(
            f"Saved splits to {self.splits_file}  ({n.get('train', 0)} train / "
            f"{n.get('validation', 0)} val / {n.get('test', 0)} test)"
        )
        return splits

    # --- examples ----------------------------------------------------------------

    def head_arity(self, class_id: str) -> int:
        return self.targets[class_id]["arity"]

    def example_row(self, example_id: str) -> str:
        return example_id.split(",", 1)[0]

    def split_examples(self, target_id, split_ids, caps):
        """A seeded random sample of the target's atoms over each split's rows, up to the caps.

        Negatives are drawn by index over the rows' non-positive candidates, without listing them.
        Each split's examples are in row order, then candidate order.
        """
        t = self.targets[target_id]
        candidates, positives = t["candidates"], t["positives"]
        out = {}
        for split, ids in split_ids.items():
            rng = random.Random(f"{SAMPLE_SEED}:{target_id}:{split}")
            max_pos, max_neg = caps.get(split, (None, None))
            rows = [r for r in self._rows(target_id) if r in ids]

            pos = [(r, i) for r in rows for i in positives.get(r, ())]
            if max_pos is not None and len(pos) > max_pos:
                pos = rng.sample(pos, max_pos)

            n_neg = [len(candidates) - len(positives.get(r, ())) for r in rows]
            ends = list(itertools.accumulate(n_neg))
            total = ends[-1] if ends else 0
            k = total if max_neg is None else min(max_neg, total)
            neg = []
            for g in rng.sample(range(total), k):
                j = bisect.bisect_right(ends, g)
                c = g - (ends[j - 1] if j else 0)
                # the c-th candidate of the row that is not positive
                for p in positives.get(rows[j], ()):
                    if p > c:
                        break
                    c += 1
                neg.append((rows[j], c))

            def ids_of(pairs):
                pairs = sorted(pairs, key=lambda p: (self._row_position[p[0]], p[1]))
                return [_example_id(r, candidates[i]) for r, i in pairs]

            out[split] = (ids_of(pos), ids_of(neg))
        return out

    def example_sizes(self) -> pd.Series:
        """Number of facts of a row."""
        return self.molecules["mol"].map(len)

    # --- encoding ----------------------------------------------------------------

    def build_background(self, rows):
        """Each game's static facts once, then every row's ``true_*``/``does_*`` facts."""
        lines, preds = [], {}
        for game in dict.fromkeys(rows["game"]):
            for pred, facts in self.statics[game].items():
                lines += [_fact(pred, args) for args in facts]
                if facts:
                    preds[pred] = len(facts[0])
        for row, facts in zip(rows.index, rows["mol"]):
            for pred, args in facts:
                lines.append(_fact(pred, (row, *args)))
                preds[pred] = len(args) + 1
        return lines, sorted(preds.items())

    @property
    def static_predicates(self) -> frozenset:
        return frozenset(p for s in self.statics.values() for p in s)

    # --- prompting ---------------------------------------------------------------

    def class_label(self, class_id: str) -> str:
        return str(class_id)

    def describe_example(self, example_id: str) -> str:
        row, *args = example_id.split(",")
        true, moves = [], []
        for pred, a in self.molecules.at[row, "mol"]:
            kind, _, name = pred.partition("_")
            if kind == "true":
                true.append(f"{name}({','.join(map(str, a))})" if name else str(a[0]))
            else:
                move = f"{name}({','.join(map(str, a[1:]))})" if name else str(a[1])
                moves.append(f"{a[0]}: {move}")
        text = f"{','.join(args) or '-'} | {' '.join(true)}"
        return text + (f" | {', '.join(moves)}" if moves else "")

    @property
    def prompt_spec(self) -> PromptSpec:
        return PromptSpec(
            system_prompt=llm_prompt.SYSTEM_PROMPT,
            existing_predicates="",
            computed_predicates="",
            example_noun="state",
            example_format="target arguments after the state | fluents true in the state | "
            "moves taken, for next targets",
            sibling_header="Other targets of the same game:",
            contrast="the negative examples (target atoms that do not hold in their state)",
            hypothesis_note="State is the state; the other head variables are the target's "
            "arguments",
        )

    def existing_predicates(self, class_id: str) -> str:
        """The target's glossary, and three facts per background predicate of its first rows."""
        t = self.targets[class_id]
        rows = self.molecules.loc[self._rows(class_id)[:20]]
        by_pred = {}
        for line in self.build_background(rows)[0]:
            by_pred.setdefault(line.split("(", 1)[0], []).append(line)
        rng = random.Random(0)
        sample = [
            line
            for lines in by_pred.values()
            for line in rng.sample(lines, k=min(3, len(lines)))
        ]
        return (
            f"{t['glossary']}\n\nSample background facts (a few per predicate, to show the "
            "constant vocabulary):\n" + "\n".join(sample)
        )

    def hypothesis_head(self, class_id: str) -> str:
        return f"{class_id}({','.join(self.targets[class_id]['head_args'])})"

    def reference_program(self, class_id: str) -> str | None:
        """The game's GDL rules defining the class head; None where they do not flatten."""
        t = self.targets[class_id]
        return GameTarget(t["game"], t["target"], self.iggp_root).reference_program(
            class_id
        )
