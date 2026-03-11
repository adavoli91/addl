import numpy as np
import torch
from typing import Tuple
from torch.utils.data import Dataset, DataLoader

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
    
def get_dataset_dataloader(X: np.ndarray|torch.Tensor, y: np.ndarray|torch.Tensor = None, batch_size: int = 128,
                           shuffle: bool = False) -> Tuple[Dataset, DataLoader]:
    '''
    Function to get a dataset and a dataloader starting from Numpy arrays or Pytorch tensors.

    Args:
        X: Array or tensor containing the values to be used as regressors.
        y: Array or tensor containing the values to be used as target.
        batch_size: Batch size.
        shuffle: Whether to shuffle data in the dataloader.

    Returns:
        dataset: Dataset obtained from `X` (and possibly `y`).
        dataloader: Dataloader obtained from `dataset`.
    '''
    X = torch.tensor(X).float()
    if y is not None:
        y = torch.tensor(y).float()
    # create dataset and dataloader
    dataset = CreateDataset(X = X, y = y)
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle)
    #
    return dataset, dataloader