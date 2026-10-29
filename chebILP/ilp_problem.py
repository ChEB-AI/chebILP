from abc import ABC

import os


def write(path, lines):
    with open(path, "w+", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _atom(pred, args):
    return f"{pred}({', '.join(str(x) for x in args) if isinstance(args, tuple) else str(args)})"


def _aleph_mode_str(name, arity, is_head):
    """Aleph mode declaration for a predicate:
    head args are all ``+any`` (input); a body pred's first arg is ``+any`` and the rest
    ``-any`` (output), giving the natural input/output chaining."""
    if is_head or arity == 1:
        args = ["+any"] * arity
    else:
        args = ["+any"] + ["-any"] * (arity - 1)
    kind = "modeh" if is_head else "modeb"
    return f":- {kind}(*, {name}({', '.join(args)}))."


def build_aleph_background(head_name, body_predicates, train_bk_lines, head_arity=1):
    """Assemble the structural Aleph ``.b`` for a class from its Popper-format inputs.

    Deliberately carries no ``:- set(...)`` or ``:- dynamic``
    directives -- those are injected at learn time (chebILP.aleph_runner)."""
    body = sorted(body_predicates)
    lines = [
        ":- use_module(library(lists)).",
        "",
        _aleph_mode_str(head_name, head_arity, True),
    ]
    lines += [_aleph_mode_str(name, arity, False) for name, arity in body]
    lines.append("")
    lines += [
        f":- determination({head_name}/{head_arity}, {name}/{arity})."
        for name, arity in body
    ]
    lines += ["", "%% ===== background knowledge ====="]
    return "\n".join(lines) + "\n" + "\n".join(train_bk_lines) + "\n"


class ILPProblem(ABC):
    """
    Abstract base class for ILP problems.
    """

    def __init__(
        self,
        head_predicate: tuple[str, int],
        pos_samples: list[tuple | str],
        neg_samples: list[tuple | str],
        bk_facts: dict[str, list[tuple | str]],
    ):
        """
        Initialize the ILP problem.

        Args:
            head_predicate: The target predicate of the ILP problem (name, arity)
            pos_samples: A list of positive samples.
            neg_samples: A list of negative samples.
            bk_facts: A dictionary of background knowledge facts (keys are predicate names)
        """
        self.head_predicate = head_predicate
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.bk_facts = bk_facts

    def body_predicates(self) -> list[tuple[str, int]]:
        preds = []
        for pred, facts in self.bk_facts.items():
            if len(facts) == 0:
                continue
            arity = len(facts[0]) if isinstance(facts[0], tuple) else 1
            assert all(
                arity == len(fact) if isinstance(fact, tuple) else arity == 1
                for fact in facts
            ), f"All facts for predicate {pred} must have the same arity."
            preds.append((pred, arity))
        return preds

    def _bk_lines(self):
        return [
            f"{_atom(pred, fact)}."
            for pred, facts in self.bk_facts.items()
            for fact in facts
        ]

    def write_popper(self, folder: str, exs_folder: str | None = None):
        """
        Write the ILP problem to exs/bias/bk files in Popper format.

        The bias leaves max_vars/max_body as placeholders for build_bias to fill in.

        Args:
            folder: The folder where the problem files will be saved.
            exs_folder: Where exs.pl goes, if not ``folder``.
        """
        head, arity = self.head_predicate
        bias_lines = [
            "%% max_vars(TODO).",
            "%% max_body(TODO).",
            "",
            f"head_pred({head}, {arity}).",
        ]
        bias_lines += [f"body_pred({pred}, {a})." for pred, a in self.body_predicates()]

        examples = [f"pos({_atom(head, s)})." for s in self.pos_samples]
        examples += [f"neg({_atom(head, s)})." for s in self.neg_samples]

        os.makedirs(folder, exist_ok=True)
        write(os.path.join(exs_folder or folder, "exs.pl"), examples)
        write(os.path.join(folder, "bias.pl"), bias_lines)
        write(os.path.join(folder, "bk.pl"), self._bk_lines())

    def write_aleph(self, folder: str, stem: str | None = None):
        """
        Write the ILP problem to .b/.f/.n files in Aleph format.

        Args:
            folder: The folder where the problem files will be saved.
            stem: File name stem (defaults to the head predicate's name).
        """
        head, arity = self.head_predicate
        path = os.path.join(folder, stem or head)
        os.makedirs(folder, exist_ok=True)
        write(path + ".f", [f"{_atom(head, s)}." for s in self.pos_samples])
        write(path + ".n", [f"{_atom(head, s)}." for s in self.neg_samples])
        with open(path + ".b", "w+", newline="\n") as f:
            f.write(
                build_aleph_background(
                    head, self.body_predicates(), self._bk_lines(), head_arity=arity
                )
            )
