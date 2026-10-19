"""Mint every problem and write it in Popper and Aleph format under ``<out_root>/<name>/``.

Usage (from repo root):
    wsl -e bash -lc '.wslvenv/bin/python -m chebILP.minted_ilp_problems.build_minted_problems'
"""
import argparse
import os

from chebILP.minted_ilp_problems.iggp import DialMinter, SwitchesMinter
from chebILP.minted_ilp_problems.trains import TrainsMinter
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
    "zendo_mint1": (TiltedChainMinter, {}),
    "zendo_mint2": (RedChainMinter, {}),
    "zendo_mint3": (ChainMinter, {}),
    "zendo_mint4": (Ascend4Minter, {}),
    "zendo_mint5": (RedTiltedTopMinter, {}),
    "zendo_mint6": (MonoChainMinter, {}),
    "trains_mint1": (TrainsMinter, {"level": 1}),
    "trains_mint2": (TrainsMinter, {"level": 2}),
    "trains_mint3": (TrainsMinter, {"level": 3}),
    "trains_mint4": (TrainsMinter, {"level": 4}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--problem_type", default=None)
    args = ap.parse_args()

    for name, (minter_cls, minter_kwargs) in MINTERS.items():
        if args.problem_type is not None and args.problem_type != minter_cls.problem_type:
            continue
        minter = minter_cls(args.out_root, **minter_kwargs)
        problem = minter.mint(args.seed)
        out_dir = os.path.join(minter.out_root, name)
        problem.write_popper(out_dir)
        problem.write_aleph(out_dir)
        print(f"{name}: {len(problem.pos_samples)} pos / {len(problem.neg_samples)} neg -> {out_dir}")


if __name__ == "__main__":
    main()
