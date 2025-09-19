import torch
from typing import Tuple
from torch.utils.data import Dataset

class CreateDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor = None) -> None:
        '''
        Function to define a pytorch dataset.

        Args:
            X: Regressor data.
            y: Target data.

        Returns: None.
        '''
        self.X = X
        self.y = y

    def __len__(self):
        '''
        Function to get data length.

        Args: None.

        Returns: None.
        '''
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Function to slice data.

        Args:
            idx: Index used for slicing.

        Returns:
            X: Regressor data sliced at `idx`.
            y: Target data sliced at `idx`.
        '''
        X = self.X[idx]
        if self.y is not None:
            y = self.y[idx]
            return X
        return X