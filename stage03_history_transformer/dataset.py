"""The Dataset returns one sample; DataLoader stacks samples into batches."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from data import POS_SCALE


class TrajectoryDataset(Dataset):
    def __init__(self, cache_dir, split):
        with np.load(Path(cache_dir) / f"{split}.npz", allow_pickle=False) as archive:
            self.arrays = {key: archive[key] for key in archive.files}

    def __len__(self):
        return len(self.arrays["scenario_id"])

    def __getitem__(self, index):
        a = self.arrays
        return {
            "scenario_id": str(a["scenario_id"][index]),
            "hist": torch.from_numpy(a["hist"][index]),
            "hist_valid": torch.from_numpy(a["hist_valid"][index]),
            "future": torch.from_numpy(a["future_m"][index] / POS_SCALE),
            "future_valid": torch.from_numpy(a["future_valid"][index]),
            "cv_m": torch.from_numpy(a["cv_m"][index]),
        }
