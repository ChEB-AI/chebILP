"""Dataset lookup by name, for the CLI and the generator scripts."""

from chebILP.preprocessing.base_dataset import BaseDataset

DATASET_NAMES = ("chebi", "rhea", "scope", "gdl")


def dataset_class(name: str) -> type[BaseDataset]:
    if name == "chebi":
        from chebILP.preprocessing.chebi.chebi_dataset import ChEBIDataset

        return ChEBIDataset
    if name == "rhea":
        from chebILP.preprocessing.reaction_processing.rhea_data import RheaDataset

        return RheaDataset
    if name == "scope":
        from chebILP.preprocessing.protein_processing.scope_data import ScopeDataset

        return ScopeDataset
    if name == "gdl":
        from chebILP.preprocessing.gdl.gdl_data import GDLDataset

        return GDLDataset
    raise ValueError(f"Unknown dataset {name!r}; choose from {DATASET_NAMES}")


def load_dataset(
    name: str,
    version: str,
    base_dir: str = "data",
    min_pos_samples: int = 25,
    three_star_only: bool = True,
) -> BaseDataset:
    """Construct a dataset and load its processed data. ``three_star_only`` applies to ChEBI only."""
    kwargs = {"three_star_only": three_star_only} if name == "chebi" else {}
    dataset = dataset_class(name)(
        version=str(version),
        base_dir=base_dir,
        min_pos_samples=min_pos_samples,
        **kwargs,
    )
    dataset.process_data()
    return dataset


def add_dataset_arguments(parser, default_version: str | None = None) -> None:
    """``--dataset``, ``--version``, ``--base_dir``, ``--min_pos_samples``, ``--include_two_star``."""
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="chebi")
    parser.add_argument(
        "--version",
        "--chebi_version",
        "-v",
        dest="version",
        default=default_version,
        required=default_version is None,
        help="Dataset version (ChEBI 251, Rhea 142, SCOPe 2.08, GDL a free label such as 1).",
    )
    parser.add_argument("--base_dir", default="data")
    parser.add_argument("--min_pos_samples", type=int, default=25)
    parser.add_argument(
        "--include_two_star",
        "-2",
        action="store_true",
        help="ChEBI only: use 2-star and 3-star molecules (default: 3-star only).",
    )


def dataset_from_args(args) -> BaseDataset:
    get = args.get if isinstance(args, dict) else lambda k, d=None: getattr(args, k, d)
    return load_dataset(
        get("dataset", "chebi"),
        get("version"),
        base_dir=get("base_dir", "data"),
        min_pos_samples=get("min_pos_samples", 25),
        three_star_only=not get("include_two_star", False),
    )
