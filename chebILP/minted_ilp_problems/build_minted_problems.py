"""Mint every problem and write it in Popper and Aleph format under ``<out_root>/<name>/``.

Usage (from repo root):
    wsl -e bash -lc '.wslvenv/bin/python -m chebILP.minted_ilp_problems.build_minted_problems'
"""
import argparse
import os

from chebILP.minted_ilp_problems.gdl_minter import gdl_minter, gdl_targets, iggp_games
from chebILP.minted_ilp_problems.iggp import DialMinter, SwitchesMinter
from chebILP.minted_ilp_problems.tictactoe import TicTacToeMinter
from chebILP.minted_ilp_problems.trains import AheadMinter, ChainedTrainsMinter, ShapeRunMinter
from chebILP.minted_ilp_problems.zendo import (Ascend4Minter, ChainMinter, MonoChainMinter, RedChainMinter,
                                               RedTiltedTopMinter, TiltedChainMinter)

MINTERS = {
    "iggp_mint1a": (DialMinter, {"n_positions": 6, "spin_steps": 3}),
    "iggp_mint1b": (DialMinter, {"n_positions": 8, "spin_steps": 4}),
    "iggp_mint1c": (DialMinter, {"n_positions": 10, "spin_steps": 5}),
    "iggp_mint1d": (DialMinter, {"n_positions": 12, "spin_steps": 5}),
    "iggp_mint2a": (SwitchesMinter, {"n_switches": 2, "hit_all": True, "sync": False}),
    "iggp_mint2b": (SwitchesMinter, {"n_switches": 3, "hit_all": False, "sync": True}),
    "iggp_mint2c": (SwitchesMinter, {"n_switches": 3, "hit_all": True, "sync": True}),
    "iggp_mint2d": (SwitchesMinter, {"n_switches": 5, "hit_all": True, "sync": True}),
    "iggp_ttt_goal": (TicTacToeMinter, {"target": "goal"}),
    "iggp_ttt_terminal": (TicTacToeMinter, {"target": "terminal"}),
    "iggp_ttt_legal_mark": (TicTacToeMinter, {"target": "legal_mark"}),
    "iggp_ttt_next_cell": (TicTacToeMinter, {"target": "next_cell"}),
    "iggp_ttt_next_control": (TicTacToeMinter, {"target": "next_control"}),
    "zendo_mint1": (TiltedChainMinter, {}),
    "zendo_mint2": (RedChainMinter, {}),
    "zendo_mint3": (ChainMinter, {}),
    "zendo_mint4": (Ascend4Minter, {}),
    "zendo_mint5": (RedTiltedTopMinter, {}),
    "zendo_mint6": (MonoChainMinter, {}),
    "trains_mint1a": (ChainedTrainsMinter, {"level": 1}),
    "trains_mint1b": (ChainedTrainsMinter, {"level": 2}),
    "trains_mint1c": (ChainedTrainsMinter, {"level": 3}),
    "trains_mint1d": (ChainedTrainsMinter, {"level": 4}),
    "trains_mint1e": (ChainedTrainsMinter, {"level": 5}),
    "trains_mint2": (AheadMinter, {}),
    "trains_mint3": (ShapeRunMinter, {}),
}

# GDL games from the IGGP repository (data/iggp), one problem per target, named gdl_<game>_<target>.
GDL_GAMES = iggp_games()
MINTERS.update({f"gdl_{g}_{t}": (gdl_minter(g), {"target": t}) for g in GDL_GAMES for t in gdl_targets(g)})


def selected(problem_type: str | None, minter_cls) -> bool:
    """Whether ``--problem_type`` (a prefix, e.g. ``gdl`` or ``gdl_tictactoe``) selects the minter."""
    return problem_type is None or minter_cls.problem_type.startswith(problem_type)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--problem_type", default=None)
    args = ap.parse_args()

    for name, (minter_cls, minter_kwargs) in MINTERS.items():
        if not selected(args.problem_type, minter_cls):
            continue
        minter = minter_cls(args.out_root, **minter_kwargs)
        problem = minter.mint(args.seed)
        out_dir = os.path.join(minter.out_root, name)
        problem.write_popper(out_dir)
        problem.write_aleph(out_dir)
        print(f"{name}: {len(problem.pos_samples)} pos / {len(problem.neg_samples)} neg -> {out_dir}")


if __name__ == "__main__":
    main()
