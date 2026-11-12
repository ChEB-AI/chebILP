"""Precompiled train BKs for Prolog (Popper's tester and Aleph).

Consulting a BK costs SWI-Prolog about 5 µs per fact; a ``.qlf`` compiled from it loads in ~1.4 s even at
9M facts. Each class's train atoms ``bk.pl`` is compiled to ``bk.qlf`` next to it. A library run loads the
same ``bk.qlf`` plus ``bk_aux.pl``, the facts of its ``bk.pl`` that the atoms one lacks, so it pays only for
what the library adds. Popper still derives its constraints from the full ``bk.pl``; Aleph's ``.b`` holds
only modes and determinations.

A ``.qlf`` loads only in the SWI-Prolog version that wrote it. One from another version loads nothing and
only prints a warning, so ``probe_goals`` gives a fact of each file to test for after loading.
"""

import os
import subprocess
import sys

from chebILP.ilp_path_manager import get_bk_path

QLF_NAME = "bk.qlf"
AUX_NAME = "bk_aux.pl"


def _fact_lines(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("%"):
                yield line


def _stale(target: str, source: str) -> bool:
    return not os.path.exists(target) or os.path.getmtime(target) < os.path.getmtime(
        source
    )


def _prolog_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def qlf_path(class_id, problem_dir) -> str:
    return os.path.join(
        os.path.dirname(get_bk_path(class_id, "train", problem_dir)), QLF_NAME
    )


def aux_path(class_id, problem_dir, aux_library_dir) -> str:
    bk = get_bk_path(class_id, "train", problem_dir, aux_library_dir)
    return os.path.join(os.path.dirname(bk), AUX_NAME)


def write_aux(class_id, problem_dir, aux_library_dir, lines=None):
    """Write the library's ``bk_aux.pl``: ``lines`` if given (the library facts as built), else the facts of
    the library ``bk.pl`` that the atoms ``bk.pl`` lacks."""
    if lines is None:
        atoms = {
            line.replace(" ", "")
            for line in _fact_lines(get_bk_path(class_id, "train", problem_dir))
        }
        lib_bk = get_bk_path(class_id, "train", problem_dir, aux_library_dir)
        lines = [
            line for line in _fact_lines(lib_bk) if line.replace(" ", "") not in atoms
        ]
    with open(aux_path(class_id, problem_dir, aux_library_dir), "w", newline="\n") as f:
        f.writelines(line + "\n" for line in lines)


class QlfCompiler:
    """Compiles train atoms BKs to ``bk.qlf`` in up to ``workers`` swipl processes at a time. It uses no
    threads: build_bk forks grounding children, and forking a multi-threaded process can deadlock."""

    def __init__(self, workers: int = 4):
        self.workers = max(1, workers)
        self.running = []
        self.failed = []

    def submit(self, class_id, problem_dir, force=False):
        """Compile the class's train atoms ``bk.pl`` if forced or its ``bk.qlf`` is missing or older."""
        bk = get_bk_path(class_id, "train", problem_dir)
        qlf = qlf_path(class_id, problem_dir)
        if not force and not _stale(qlf, bk):
            return
        while len(self.running) >= self.workers:
            self._wait_one()
        if os.path.exists(qlf):
            os.remove(qlf)
        proc = subprocess.Popen(
            ["swipl", "-q", "-g", f"qcompile('{_prolog_path(bk)}')", "-t", "halt"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.running.append((class_id, proc))

    def _wait_one(self):
        class_id, proc = self.running.pop(0)
        _, err = proc.communicate()
        if proc.returncode != 0:
            self.failed.append((class_id, err.strip()[-500:]))

    def finish(self) -> list:
        """Wait for all compiles; returns ``[(class_id, error)]`` of those that failed."""
        while self.running:
            self._wait_one()
        return self.failed


def prepare_prolog_bks(class_ids, problem_dir, aux_library_dir=None, workers=4):
    """For problem dirs built earlier: compile each class's train atoms ``bk.qlf`` and, with a library,
    write its ``bk_aux.pl``, each only when missing or older than its source."""
    missing = [
        c for c in class_ids if not os.path.exists(get_bk_path(c, "train", problem_dir))
    ]
    if missing:
        raise FileNotFoundError(
            f"No train atoms bk.pl for {len(missing)} class(es): {', '.join(missing[:10])}"
        )
    compiler = QlfCompiler(workers)
    for c in class_ids:
        compiler.submit(c, problem_dir)
        if aux_library_dir:
            lib_bk = get_bk_path(c, "train", problem_dir, aux_library_dir)
            if os.path.exists(lib_bk) and _stale(
                aux_path(c, problem_dir, aux_library_dir), lib_bk
            ):
                write_aux(c, problem_dir, aux_library_dir)
    failed = compiler.finish()
    if failed:
        raise RuntimeError(f"qcompile failed for {len(failed)} class(es): {failed[:3]}")


def prolog_bk_files(
    class_id, problem_dir, aux_library_dir=None, verify=True
) -> list[str]:
    """Files Prolog loads for the class's train BK: ``bk.qlf``, and ``bk_aux.pl`` with a library. Raises
    if one is missing or older than its source, or, with ``verify``, if the library ``bk.pl`` is not the
    atoms one plus ``bk_aux.pl`` (e.g. one of the two BKs rebuilt from different examples)."""
    atoms_bk = get_bk_path(class_id, "train", problem_dir)
    qlf = qlf_path(class_id, problem_dir)
    if _stale(qlf, atoms_bk):
        raise FileNotFoundError(
            f"{qlf} is missing or older than {atoms_bk}; run compile_bk (or build_bk) first."
        )
    files = [qlf]
    if aux_library_dir:
        lib_bk = get_bk_path(class_id, "train", problem_dir, aux_library_dir)
        aux = aux_path(class_id, problem_dir, aux_library_dir)
        if _stale(aux, lib_bk):
            raise FileNotFoundError(
                f"{aux} is missing or older than {lib_bk}; run compile_bk (or build_bk) with "
                "this --predicate_dir first."
            )
        if verify:
            n_lib, n_atoms, n_aux = (
                sum(1 for _ in _fact_lines(p)) for p in (lib_bk, atoms_bk, aux)
            )
            if n_lib != n_atoms + n_aux:
                raise ValueError(
                    f"{lib_bk} has {n_lib} facts, but {atoms_bk} and {aux} have {n_atoms} + {n_aux}; "
                    "rebuild both BKs from the same examples."
                )
        files.append(aux)
    return [_prolog_path(p) for p in files]


def probe_goals(files: list[str]) -> list[str]:
    """One fact per file of ``prolog_bk_files``, as a goal that holds once the file has loaded: the first
    fact of the ``bk.pl`` a ``.qlf`` was compiled from, else the file's own first fact."""
    goals = []
    for f in files:
        source = f[: -len(".qlf")] + ".pl" if f.endswith(".qlf") else f
        first = next(_fact_lines(source), None)
        if first:
            goals.append(first.rstrip("."))
    return goals


def check_loadable(files: list[str], tool: str):
    """Raise unless ``files`` load in the Prolog that ``tool`` uses, janus (Popper) or swipl (Aleph), with
    their facts present."""
    goals = probe_goals(files)
    if tool == "aleph":
        load = ",".join(f"load_files('{f}',[])" for f in files)
        probe = ",".join(f"once({g})" for g in goals) or "true"
        goal = f"catch(({load},{probe}),E,(print_message(error,E),halt(1)))"
        cmd = ["swipl", "-q", "-g", goal, "-t", "halt"]
    else:
        script = (
            "from janus_swi import consult, query_once\n"
            f"for f in {files!r}: consult(f)\n"
            f"for g in {goals!r}: assert query_once(g)['truth'], g\n"
        )
        cmd = [sys.executable, "-c", script]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        output = (result.stderr + result.stdout).strip()[-500:]
        raise RuntimeError(
            f"{files[0]} does not load in the Prolog {tool} uses (a .qlf loads only in the SWI-Prolog "
            f"version that wrote it; rerun compile_bk): {output}"
        )
