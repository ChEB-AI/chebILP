
from typing import Literal
import sys
from contextlib import contextmanager
from datetime import datetime

AVAILABLE_PREDICATE_SETS = Literal["atoms", "chembl_fgs", "chebi_fgs", "chebi_fg_rules", "chebi_fg_learned_rules", "llm_generated_fgs", "llm_generated_rules", "farm_fgs", "farm_fgs_atoms", "fowl"]


def split_prolog_literals(body):
    """Split a Prolog rule body into literals, respecting parenthesis depth."""
    literals, current, depth = [], [], 0
    curly_depth = 0
    for char in body:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == '{':
            curly_depth += 1
            current.append(char)
        elif char == '}':
            curly_depth -= 1
            current.append(char)
        elif char == ',' and depth == 0 and curly_depth == 0:
            literals.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        literals.append(''.join(current).strip())
    return literals


@contextmanager
def tee_output(log_path):
    """Tee stdout/stderr to a log file while preserving console output."""
    log_file = open(log_path, "a", encoding="utf-8")

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
            self._buffer = ""

        def _emit(self, line, end=""):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for stream in self._streams:
                stream.write(f"[{timestamp}] {line}{end}")

        def write(self, data):
            self._buffer += data
            while True:
                newline_index = self._buffer.find("\n")
                if newline_index == -1:
                    break
                line = self._buffer[:newline_index].rstrip("\r")
                self._buffer = self._buffer[newline_index + 1:]
                self._emit(line, "\n")

        def flush(self):
            if self._buffer:
                self._emit(self._buffer)
                self._buffer = ""
            for stream in self._streams:
                stream.flush()

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()


def get_atom_id(atom: int, molecule_id):
    return "a" + str(molecule_id) + "_" + str(atom + 1)  # Prolog indices start at 1


def sort_labels_by_hierarchy(labels: list[str], chebi_graph: nx.DiGraph):
    """Sort a list of ChEBI ids by their hierarchy in ChEBI. If A is a superclass of B, A will appear before B in the returned list."""
    if any(label not in chebi_graph for label in labels):
        raise ValueError(f"Some labels are not in the ChEBI graph: {set(labels) - set(chebi_graph)}")
    label_set = set(labels)

    order_graph = nx.DiGraph()
    order_graph.add_nodes_from(labels)
    for label in labels:
        for superclass in nx.descendants(chebi_graph, label) & label_set:
            order_graph.add_edge(superclass, label)  # superclass -> subclass

    return list(nx.topological_sort(order_graph))