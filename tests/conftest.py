import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from braindecode.augmentation import IdentityTransform
from braindecode.augmentation import SignFlip
from torch.utils.data import Dataset


@pytest.fixture(scope="module")
def random_state():
    return np.random.randint(0, 1e5)


@pytest.fixture(scope='module')
def dummy_prediction():
    pred_list = []
    for i in range(10):
        pred_list.append({
            'y_pred': np.random.randint(4, size=100),
            'y_true': np.random.randint(4, size=100),
            'set': 'test',
            'fold': i,
            'proportion': 1,
            'augmentation': str(IdentityTransform())
        })
        pred_list.append({
            'y_pred': np.random.randint(4, size=100),
            'y_true': np.random.randint(4, size=100),
            'set': 'test',
            'fold': i,
            'proportion': 1,
            'augmentation': str(SignFlip(0.5))
        })
    return pd.DataFrame(pred_list)


class MockDataset(Dataset):
    def __init__(self, n_samples, n_channels=22) -> None:
        self.X = torch.Tensor(np.random.normal(
            size=(n_samples, n_channels, 1000)
        ))
        self.y = torch.Tensor(np.random.randint(5, size=n_samples))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@pytest.fixture(scope='module')
def mock_dataset():
    return MockDataset(50)


@pytest.fixture(scope='session')
def cachedir():
    return tempfile.TemporaryDirectory().name
