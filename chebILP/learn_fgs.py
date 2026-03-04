
import os
import pandas as pd

from chebILP.ilp_path_manager import get_exs_path
from chebILP.mol2ilp import ILPProblemBuilder

# todo: this class is out of dat
class FGILPProblemBuilder(ILPProblemBuilder):

    @property
    def problem_dir(self):
        return self._problem_dir if self._problem_dir else os.path.join("ilp", "learn_fgs")
    
    def load_samples(self, dataset_path):
        if dataset_path is None:
            dataset_path = os.path.join("data", "chebi_fgs_dataset.pkl")
        return pd.read_pickle(dataset_path)


    def gather_samples_for_chebi_cls(self, target_id, max_pos_samples=100, max_neg_samples=100) -> pd.DataFrame:
        train_samples_df = self.samples_df[[str(id) in self.train_ids for id in self.samples_df.index]]

        df_pos = train_samples_df[train_samples_df[f"has_part_{target_id}"]]
        df_neg = train_samples_df[~train_samples_df[f"has_part_{target_id}"]]

        pos_samples = df_pos.sample(min(max_pos_samples, len(df_pos)))

        neg_samples = self.get_closest_negatives(df_neg, target_id, n_samples=max_neg_samples)
        
        exs_path = get_exs_path(target_id, base_dir=self.problem_dir, split="train")
        with open(exs_path, "w+") as f:
            for sample in pos_samples.index:
                f.write(f"pos(chebi_{target_id}({sample})).\n")
            for sample in neg_samples.index:
                f.write(f"neg(chebi_{target_id}({sample})).\n")

        print(f"Training on {len(pos_samples)} positive and {len(neg_samples)} negative samples")

        return pd.concat([pos_samples, neg_samples])

    def gather_validation_samples(self, target_id, validation_samples_df, max_pos_samples=100, max_neg_samples=100) -> tuple[int, int]:

        df_pos = validation_samples_df[validation_samples_df[f"has_part_{target_id}"]]
        df_neg = validation_samples_df[~validation_samples_df[f"has_part_{target_id}"]]

        pos_samples = df_pos.sample(min(max_pos_samples, len(df_pos)))

        neg_samples = self.get_closest_negatives(df_neg, target_id, n_samples=max_neg_samples)
        
        exs_validation_path = get_exs_path(target_id, base_dir=self.problem_dir, split="validation")
        with open(exs_validation_path, "w+") as f:
            for sample in pos_samples.index:
                f.write(f"pos(chebi_{target_id}({sample})).\n")
            for sample in neg_samples.index:
                f.write(f"neg(chebi_{target_id}({sample})).\n")

        return len(df_pos), len(df_neg)