"""Run Aleph on a chebILP training problem, plain or seeded, as a Popper alternative.

The problem's structural Aleph ``.b`` (modes + determinations + train bk) and its ``.f``/``.n``
example files are produced at build time (``ilp_problem_builder``). Here they are turned into a
runnable program by injecting the learn-time search settings (and, when seeding, a user
refinement graph built from the LLM's class hypothesis), then handed to the vendored engine
``chebILP/aleph/aleph.pl`` via ``run_aleph.pl`` under ``swipl``.

``run_ilp_training_aleph`` returns the same shape as ``run_ilp_training_subprocess`` so
``learn_chebi_classes`` can branch on the tool with no change downstream:
``{"prog_str", "score":[tp,fn,tn,fp], "num_programs", "time_to_best"}``.

The clause-parsing / hypothesis-unit helpers are ported from the ``scripts/_aleph_*`` prototype.
"""
import itertools
import os
import re
import shutil
import subprocess

ALEPH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aleph")
RUN_ALEPH_PL = os.path.join(ALEPH_DIR, "run_aleph.pl")

# Aleph search defaults (from the scripts/_aleph_driver.py prototype). clauselength is mapped
# from the learn-time max_body; searchtime bounds each clause's search so a long run stops
# cleanly and still prints its best theory (rather than being SIGKILLed on subprocess timeout).
_DEFAULT_SETTINGS = {"i": 3, "noise": 200, "minpos": 2, "nodes": 50000, "verbosity": 0}
_SUBPROCESS_TIMEOUT_BUFFER = 60  # extra wall-clock over the internal budget, for saturation + IO
_MAX_SEED_UNITS = 6  # above this, seed with the full hypothesis only (avoid 2^n start nodes)

_BODY_PRED_RE = re.compile(r"body_pred\(\s*([a-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)")
_VAR_RE = re.compile(r"^[A-Z_][A-Za-z0-9_]*$")


# ── Prolog-clause parsing (ported from scripts/_aleph_driver.py) ─────────────────────────
def split_conj(text):
    """Split a Prolog conjunction on top-level commas (respecting nested parens)."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def literal_pred(lit):
    return lit.split("(", 1)[0].strip()


def literal_args(lit):
    inside = lit[lit.index("(") + 1: lit.rindex(")")]
    return split_conj(inside)


def literal_vars(lit):
    return [a for a in literal_args(lit) if _VAR_RE.match(a)]


def clause_body_literals(clause):
    return split_conj(clause.split(":-", 1)[1].strip().rstrip("."))


def hypothesis_units(clause, mol_var="A"):
    """Split a hypothesis body into droppable units, each a list of literal strings.

    A unit is a connected component of the body's atom variables (sharing a variable directly
    or transitively through a link like ``has_bond_to``), together with the ``has_atom``
    literals binding those atoms to the molecule. A literal mentioning only the molecule
    variable is its own unit. Any subset of units stays range-restricted."""
    lits = clause_body_literals(clause)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    lit_atomvars = []
    for lit in lits:
        vs = [v for v in literal_vars(lit) if v != mol_var]
        lit_atomvars.append(vs)
        for v in vs[1:]:
            union(vs[0], v)

    units, index_of = [], {}
    for lit, vs in zip(lits, lit_atomvars):
        if not vs:
            units.append([lit])
            continue
        root = find(vs[0])
        if root not in index_of:
            index_of[root] = len(units)
            units.append([])
        units[index_of[root]].append(lit)
    return units


# ── Runnable-.b assembly ─────────────────────────────────────────────────────────────────
def _dynamic_decls(head_name, body_preds):
    """``:- dynamic p/a.`` for the head and every body pred, so a pred declared in the bias
    but with zero facts in bk.pl fails gracefully instead of throwing existence_error in SWI."""
    decls = [f"{head_name}/1"] + [f"{n}/{a}" for n, a in sorted(body_preds)]
    return "".join(f":- dynamic {d}.\n" for d in decls)


def _settings_lines(settings):
    return "".join(f":- set({k}, {v}).\n" for k, v in settings.items())


def _refine_block(head_name, units, body_preds):
    """User refinement graph: start nodes are every non-empty subset of the hypothesis units
    (the full seed and its generalizations), refined by adding one so-far-unused unary
    qualifier on any existing variable. Ported from _aleph_driver.write_seed_b."""
    head = f"{head_name}(A)"
    start_lines = []
    n = len(units)
    if n > _MAX_SEED_UNITS:
        # Subsets grow as 2^n; for a seed with many disjoint units, fall back to the full
        # seed as the sole start node (still specialised below) rather than 2^n-1 start facts.
        start_lines.append(f"refine(false, ({head} :- {', '.join(l for u in units for l in u)})).")
    else:
        for r in range(n, 0, -1):  # largest (full seed) first
            for combo in itertools.combinations(range(n), r):
                lits = [l for i in combo for l in units[i]]
                start_lines.append(f"refine(false, ({head} :- {', '.join(lits)})).")

    seed_names = {literal_pred(l) for u in units for l in u} | {"has_atom"}
    qual = sorted(name for (name, ar) in body_preds if ar == 1 and name not in seed_names)
    qual_facts = "".join(f"unary_pred({q}).\n" for q in qual)

    return f"""
%% ===== seeded search: LLM-hypothesis start nodes + neighbourhood =====
{qual_facts}
%% start nodes (full hypothesis first, then its generalizations)
{chr(10).join(start_lines)}

%% specialise: add one so-far-unused unary qualifier on any existing variable
refine(({head} :- Body), ({head} :- NewBody)) :-
    body_var(Body, V),
    unary_pred(P),
    Lit =.. [P, V],
    \\+ conj_has(Lit, Body),
    conj_snoc(Body, Lit, NewBody).

body_var(Body, V) :- term_variables(Body, Vs), member(V, Vs).

conj_has(L, (A,_)) :- L == A, !.
conj_has(L, (_,B)) :- !, conj_has(L, B).
conj_has(L, A)     :- L == A.

conj_snoc((A,B), Lit, (A,NB)) :- !, conj_snoc(B, Lit, NB).
conj_snoc(A, Lit, (A,Lit)).
"""


def _assemble_runnable_b(structural_b, head_name, body_preds, settings, seed_clause):
    """Inject dynamic decls + settings (+ seed refine graph) into the structural .b text."""
    settings = dict(settings)
    if seed_clause:
        settings["refine"] = "user"
    pre = "\n" + _dynamic_decls(head_name, body_preds) + "\n" + _settings_lines(settings)
    anchor = ":- use_module(library(lists)).\n"
    body = structural_b.replace(anchor, anchor + pre, 1)
    if seed_clause:
        units = hypothesis_units(seed_clause)
        if units:
            body += _refine_block(head_name, units, body_preds)
    return body


# ── Output parsing ───────────────────────────────────────────────────────────────────────
def _normalize_clause(clause_text):
    return re.sub(r"\s+", " ", clause_text).strip().rstrip(".") + "."


def parse_aleph_output(text, chebi_id):
    """Extract the learned program, train score, clause count and time-to-best from the
    run_aleph.pl output. Returns the run_ilp_training dict shape."""
    def last_num(pat, cast=float):
        vals = re.findall(pat, text)
        return cast(vals[-1]) if vals else None

    num_programs = last_num(r"\[total clauses constructed\] \[(\d+)\]", int)
    if num_programs is None:
        num_programs = last_num(r"\[clauses constructed\] \[(\d+)\]", int)
    time_to_best = last_num(r"=== TIME_TO_BEST: ([\d.]+) s ===")

    # Program: clauses from the final theory section (empty if the run found none).
    prog_str = None
    fi = text.find("=== FINAL THEORY ===")
    if fi != -1:
        seg = text[fi:]
        bodies = re.findall(r"(chebi_\d+\(A\) :-\s*\n(?:\s+.*\n)*?)(?=\n\[|\n\n|\nchebi_)", seg)
        clauses = [_normalize_clause(b) for b in bodies if ":-" in b]
        if clauses:
            prog_str = "\n".join(clauses)
    if prog_str is None:
        # Fallback: the last best-so-far clause, which survives even a search cut short.
        found = re.findall(
            r"\[found clause\]\s*\n(chebi_\d+\(A\)[^\[]*?)\n\[pos cover = \d+ neg cover = \d+\]",
            text)
        if found:
            prog_str = _normalize_clause(found[-1])

    # Score: Aleph's [Training set summary] [[TP,FP,FN,TN]] -> Popper's [tp,fn,tn,fp].
    # Only meaningful alongside a program; an empty theory prints an all-negative summary,
    # which would otherwise masquerade as a real (TP=0) score, so drop it like Popper's None.
    score = None
    if prog_str is not None:
        m = re.search(r"\[Training set summary\] \[\[(\d+),(\d+),(\d+),(\d+)\]\]", text)
        if m:
            tp, fp, fn, tn = (int(x) for x in m.groups())
            score = [tp, fn, tn, fp]

    return {"prog_str": prog_str, "score": score,
            "num_programs": num_programs, "time_to_best": time_to_best}


# ── Entry point ──────────────────────────────────────────────────────────────────────────
def run_ilp_training_aleph(chebi_id, aleph_stem, bias_path, timeout, max_body=8,
                           seed_clause=None, log_dir=None, settings_overrides=None):
    """Learn one class with Aleph. Mirrors run_ilp_training_subprocess's return contract.

    ``aleph_stem`` is the build-time stem (``get_aleph_stem``); ``<stem>.b/.f/.n`` must exist.
    The runnable .b (settings + optional seed graph injected) and the copied .f/.n are written
    under ``log_dir/aleph/`` so a run is reproducible and inspectable."""
    from chebILP.ilp_classifier import log_subprocess_output

    with open(bias_path) as f:
        body_preds = {(m.group(1), int(m.group(2))) for m in _BODY_PRED_RE.finditer(f.read())}
    with open(aleph_stem + ".b") as f:
        structural_b = f.read()

    settings = dict(_DEFAULT_SETTINGS)
    settings["clauselength"] = max_body + 1
    settings["searchtime"] = timeout
    if seed_clause:
        settings["explore"] = "true"
    if settings_overrides:
        settings.update(settings_overrides)

    head_name = f"chebi_{chebi_id}"
    runnable_b = _assemble_runnable_b(structural_b, head_name, body_preds, settings, seed_clause)

    run_dir = os.path.join(log_dir or ".", "aleph")
    os.makedirs(run_dir, exist_ok=True)
    run_stem = os.path.join(run_dir, str(chebi_id))
    with open(run_stem + ".b", "w") as f:
        f.write(runnable_b)
    for ext in (".f", ".n"):
        shutil.copyfile(aleph_stem + ext, run_stem + ext)

    swipl_stem = os.path.abspath(run_stem).replace("\\", "/")
    # aleph.pl bounds the whole greedy induce by searchtime (= timeout), but that guard only
    # fires between clauses, so the last clause can start near the budget and run ~timeout more.
    # The outer wall-clock cap must exceed that (~2x) so the internal guard ends the run cleanly
    # with a printed theory, rather than a SIGKILL mid-search.
    hard_timeout = 2 * timeout + _SUBPROCESS_TIMEOUT_BUFFER
    try:
        result = subprocess.run(
            ["swipl", "-q", "-g", f"main('{swipl_stem}')", "-t", "halt", RUN_ALEPH_PL],
            capture_output=True, text=True, cwd=ALEPH_DIR, timeout=hard_timeout,
        )
        out = (result.stdout or "") + "\n" + (result.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or "") if isinstance(e.stdout, str) else (e.stdout.decode() if e.stdout else "")) \
            + "\n" + ((e.stderr or "") if isinstance(e.stderr, str) else (e.stderr.decode() if e.stderr else ""))

    with open(run_stem + ".out", "w") as f:
        f.write(out)
    if log_dir:
        log_subprocess_output(log_dir, f"Aleph training: {run_stem}.b", out)
    return parse_aleph_output(out, chebi_id)
