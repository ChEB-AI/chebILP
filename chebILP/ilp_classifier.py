from popper.loop import learn_solution, get_bk_cons, timeout, popper
from popper.tester import Tester, format_literal_janus, deduce_neg_example_recalls
from popper.util import Settings, format_prog, Literal
import os
import subprocess
import sys
import json
import pickle
import base64
from datetime import datetime
import time
import re
from janus_swi import consult, query_once
from bitarray.util import ones

def log_subprocess_output(log_dir, phase, result):
    """Write subprocess stdout/stderr to the run log with timestamp."""
    if not log_dir:
        return
    log_file = os.path.join(log_dir, "run.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(result, str):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[{timestamp}] === {phase} ===\n")
            for line in result.splitlines():
                f.write(f"[{timestamp}] {line}\n")
        return
    if not result.stdout.strip() and not result.stderr.strip():
        return
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] === {phase} (Return code: {result.returncode}) ===\n")
        if result.stdout.strip():
            f.write("--- stdout ---\n")
            for line in result.stdout.splitlines():
                f.write(f"[{timestamp}] [stdout] {line}\n")
        if result.stderr.strip():
            f.write("--- stderr ---\n")
            for line in result.stderr.splitlines():
                f.write(f"[{timestamp}] [stderr] {line}\n")


def reload_tester(tester, settings):
        exs_pl_path = settings.ex_file

        if not settings.pi_enabled:
            consult('prog', f':- dynamic {settings.head_literal.predicate}/{len(settings.head_literal.arguments)}.')

        for x in [exs_pl_path]:
            if os.name == 'nt': # if on Windows, SWI requires escaped directory separators
                x = x.replace('\\', '\\\\')
            consult(x)

        query_once('load_examples')

        neg_literal = Literal('neg_fact', tuple(range(len(settings.head_literal.arguments))))
        tester.neg_fact_str = format_literal_janus(neg_literal)
        tester.neg_literal_set = frozenset([neg_literal])

        q = 'findall(_Atom2, (neg_index(_K, _Atom1), term_string(_Atom1, _Atom2)), S)'
        res = query_once(q)['S']
        atoms = []
        for x in res:
            x = x[:-1].split('(')[1].split(',')
            atoms.append(x)

        if atoms:
            try:
                settings.recall = settings.recall | deduce_neg_example_recalls(settings, atoms)
            except Exception as e:
                print(e)

        tester.num_pos = query_once('findall(_K, pos_index(_K, _Atom), _S), length(_S, N)')['N']
        tester.num_neg = query_once('findall(_K, neg_index(_K, _Atom), _S), length(_S, N)')['N']

        print(f"Reloaded tester with {tester.num_pos} positive and {tester.num_neg} negative examples.")

        tester.pos_examples_ = ones(tester.num_pos)

        tester.cached_pos_covered = {}
        tester.cached_inconsistent = {}

        if tester.settings.recursion_enabled:
            query_once(f'assert(timeout({tester.settings.eval_timeout})), fail')


class PopperWrapper:

    def __init__(self, settings_parameters):
        # override default settings for Popper
        self.settings_parameters = {
            "noisy": True,
            "anytime_solver": "nuwls",
        }
        self.settings_parameters.update(settings_parameters)
        self.settings = None
        self.tester = None

    def solve(self, chebi_id, exs_file, bk_file, bias_file):
        if self.settings is None:
            self.settings = Settings(ex_file=exs_file, bk_file=bk_file, bias_file=bias_file, **self.settings_parameters)
            self.settings.nonoise = not self.settings.noisy
            self.settings.datalog = False

        else:
            # override head_pred
            self.settings.head_pred = f"chebi_{chebi_id}"
            # Clear cache
            from janus_swi import query_once
            query_once("abolish(pos/1), true")
            query_once("(retractall(pos_index(_, _)) ; abolish(pos_index/2), true)")
            query_once("abolish(neg/1), true")
            query_once("(retractall(neg_index(_, _)) ; abolish(neg_index/2), true)")
            query_once("(retractall(neg_fact(_, _)) ; abolish(neg_fact/2), true)")
            num_pos = query_once('findall(_K, pos_index(_K, _Atom), _S), length(_S, N)')['N']
            num_neg = query_once('findall(_K, neg_index(_K, _Atom), _S), length(_S, N)')['N']
            assert num_pos == 0 and num_neg == 0, f"Cache not cleared properly: {num_pos} positive and {num_neg} negative examples remain."

            # set ex_file 
            self.settings.ex_file = exs_file
            self.settings.bk_file = bk_file
            self.settings.bias_file = bias_file

        with self.settings.stats.duration('load data'):
            self.tester = Tester(self.settings)
        # learn_solution
        self.settings.solution_found = False
        self.settings.solution = None
        self.settings.best_prog_score = None
        bkcons = get_bk_cons(self.settings, self.tester)
        self.settings.datalog = False
        timeout(self.settings, popper, (self.settings, self.tester, bkcons), timeout_duration=int(self.settings.timeout),)
        prog_str = format_prog(self.settings.solution) if self.settings.solution else None
        return {
            "prog": self.settings.solution, 
            "prog_str": prog_str, 
            "score": list(self.settings.best_prog_score) if self.settings.best_prog_score else None
            }
    

    
def run_ilp_training_subprocess(exs_file, bk_file, bias_file, settings_parameters, log_dir=None):
    """Run Popper ILP learning in a separate subprocess for isolated Prolog session."""
    #print(f"Running ILP training subprocess with exs_file={exs_file}, bk_file={bk_file}, bias_file={bias_file}...")
    #print(f"Settings parameters: {settings_parameters}")
    script = f'''
import json
import pickle
import base64
from popper.loop import learn_solution
from popper.util import Settings, format_prog

settings = Settings(ex_file=r"{exs_file}", bk_file=r"{bk_file}", bias_file=r"{bias_file}", **{repr(settings_parameters)})
prog, score, stats = learn_solution(settings)
prog_str = format_prog(prog) if prog else None

result = {{"prog_str": prog_str, "score": list(score) if score else None}}
print(json.dumps(result))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        start_new_session=True,  # Start in a new session to isolate from parent process
        cwd=os.getcwd(),
    )
    if log_dir:
        log_subprocess_output(log_dir, f"Training: {bias_file}", result)
    # Parse only the last line (JSON output), ignore earlier lines (warnings/progress)
    stdout_lines = result.stdout.strip().split('\n')
    try:
        output = json.loads(stdout_lines[-1])
    except json.decoder.JSONDecodeError:
        output = {"prog_str": None, "score": None}
        print(f"    Failed to parse JSON output. See logs for details.")
    
    return output


def split_prolog_literals(body):
    """Split Prolog body into literals by comma, respecting parentheses depth."""
    literals = []
    current = []
    depth = 0
    
    for char in body:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            # Comma at depth 0 - this is a literal separator
            literals.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    
    # Don't forget the last literal
    if current:
        literals.append(''.join(current).strip())
    
    return literals


def format_literal(literal_str):
    predicate = literal_str.split('(')[0].strip()
    args_str = literal_str.split('(')[1].rstrip(')').strip()
    args = [f"_{arg}" for arg in args_str.split(',')]
    return f"{predicate}({', '.join(args)})"

def run_ilp_validation(chebi_id, rule, exs_file, bk_file):
    # exs_file contains a list of pos(chebi_{chebi_id}(sample_id)) and neg(chebi_{chebi_id}(sample_id))
    # bk_file contains the same background knowledge used for training (e.g. has_atom(sample_id, atom))
    # rule has format "chebi_{chebi_id}(V0) :- body_literal1(V0, V1), body_literal2(V1), ..."   
    consult(bk_file)
    consult(exs_file)
    test_pl_path = os.path.join("data", "test.pl")
    consult(test_pl_path)
    query_once('load_examples')

    n_pos = query_once(f"findall(_ID, (pos_index(_ID, chebi_{chebi_id}(_V0))), S), length(S, N).")["N"]
    n_neg = query_once(f"findall(_ID, (neg_index(_ID, chebi_{chebi_id}(_V0))), S), length(S, N).")["N"]

    pos_covered, neg_covered = set(), set()
    # Assert each clause separately to avoid Prolog syntax errors on multi-line rules.
    clauses = [c.strip() for c in rule.replace("\r", "").split(".") if c.strip()]
    for clause in clauses:
        head, body = clause.split(":-")
        # split pred1(V0, V1), pred2(V1) into separate literals and format each with format_literal
        body = ",".join([format_literal(b) for b in split_prolog_literals(body)])
        pos = query_once(f"findall(_ID, (pos_index(_ID, chebi_{chebi_id}(_V0)), {body}), S).")["S"]
        pos_covered.update(pos)
        neg = query_once(f"findall(_ID, (neg_index(_ID, chebi_{chebi_id}(_V0)), {body}), S).")["S"]
        neg_covered.update(neg)

    tps = len(pos_covered)
    fps = len(neg_covered)
    fns = n_pos - tps
    tns = n_neg - fps
    return {
        "TP": tps,
        "FP": fps,
        "TN": tns,
        "FN": fns,
    }

def run_ilp_validation_subprocess(chebi_id, rule, exs_file, bk_file, log_dir=None, timeout=300):
    """Run ILP validation in a separate subprocess to isolate Prolog session."""
    script = f'''

import json
from janus_swi import consult, query_once
from chebILP.ilp_classifier import run_ilp_validation
res = run_ilp_validation("{chebi_id}", """{rule}""", r"{exs_file}", r"{bk_file}")

print(json.dumps(res))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        start_new_session=True,  # Start in a new session to isolate from parent process
        cwd=os.getcwd(),
        timeout=timeout,
    )
    if log_dir:
        log_subprocess_output(log_dir, f"Validation: {chebi_id}", result)
    # Parse only the last line (JSON output), ignore earlier lines (warnings/progress)
    stdout_lines = result.stdout.strip().split('\n')
    try:
        output = json.loads(stdout_lines[-1])
    except json.decoder.JSONDecodeError:
        output = {"error": "Failed to parse JSON output"}
    return output


if __name__ == "__main__":
    import json
    import pickle
    import base64
    from popper.loop import learn_solution
    from popper.util import Settings, format_prog

    settings_parameters = {
        "noisy": True,
        "anytime_solver": "nuwls",
    }
    ex_file = os.path.join("ilp", "chebi_v244", "chebi_73754", "train", "exs.pl")
    bk_file = os.path.join("ilp", "chebi_v244", "chebi_73754", "train", "chebi_fg_learned_rules", "bk.pl")
    bias_file = os.path.join("ilp", "chebi_v244", "chebi_73754", "train", "chebi_fg_learned_rules", "bias_max_vars=6_max_body=8_max_clauses=2.pl")
    settings = Settings(ex_file=f"{ex_file}", bk_file=f"{bk_file}", bias_file=f"{bias_file}", **settings_parameters)
    prog, score, stats = learn_solution(settings)
    prog_str = format_prog(prog) if prog else None

    result = {"prog_str": prog_str, "score": list(score) if score else None}
    print(result)
