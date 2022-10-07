from copy import Error

import numpy as np
import torch
from braindecode.augmentation import Transform
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing.preprocess import Preprocessor
from braindecode.preprocessing.preprocess import preprocess
from braindecode.util import set_random_seeds
from sklearn.preprocessing import scale
from torch.utils.data.dataset import Subset


class EEGDataset(BaseConcatDataset):

    """
    Subclass of BaseConcatDataset. Implements methods that cover most of the
    preprocessing pipeline. e.g. applying preprocessings such as
    standardization, extracting windows from the datastet.

    Parameters
    ----------
    dataset: BaseConcatDataset

    """

    def __init__(
            self,
            dataset):
        super().__init__(dataset.datasets)

    def __len__(self):
        return super().__len__()

    def preprocess(self, preprocessors=[
            Preprocessor(lambda x: x * 1e6),
            Preprocessor('filter', l_freq=None, h_freq=30),
            Preprocessor(fn=scale)]):
        """Preprocess the dataset using braindecode.preprocessing.preprocess
        The default preprocessing consist in converting the voltage of EEG
        channels to µV, low pass filtering the channel with a cut-off frequency
        of 30Hz and standartizing the channels

        Parameters
        ----------
        preprocessors: list
            list of preprocess operation applied to the dataset. Must be
            instances of braindecode.preprocessing.preprocess.Preprocessor
        """
        preprocess(self, preprocessors)

    def __getitem__(self, idx):
        """Remove the last element from EEG datasets. It helps to locate each
        window in the recording but is useless in our usecase.
        """
        return super().__getitem__(idx)[:2]

    def get_windows(self,
                    mapping=None,
                    window_size_samples=None,
                    window_stride_samples=None,
                    trial_start_offset_samples=0,
                    trial_stop_offset_samples=0,
                    preload=True,
                    n_jobs=1,
                    **kwargs):
        """ Extract the epochs from the EEGDataset using
        braindecode.preprocessing.create_windows_from_events

        Parameters
        ----------
        mapping: dict
            Dictionary that maps comprehensible labels to numbers.
            e.g. mapping = {'Sleep stage W': 0}
        window_size_samples: int
            Number of samples per window. Mandatory if the dataset does not
            have a stim channel.
        window_stride_samples: int
            Stride use to exctract windows from the recordings. Mandatory if
            the dataset does not have a stim channel.
        trial_start_offset_samples: int
            Start offset from original trial onsets, in samples. Defaults to
            zero.
        trial_stop_offset_samples: int
            Stop offset from original trial stop, in samples. Defaults to zero.
        preload: bool
            If True, preload the data of the Epochs objects. This is useful to
            reduce disk reading overhead when returning windows in a training
            scenario, however very large data might not fit into memory.
        n_jobs: int
            Number of jobs to use to parallelize the windowing.

        Returns
        -------
        XXX

        Note:
        -----
        The mapping can be found using `windows.datasets[0].windows.event_id`
        (not that intuitive to find)
        """
        windows = create_windows_from_events(
            self,
            mapping=mapping,
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=trial_stop_offset_samples,
            preload=preload,
            n_jobs=n_jobs,
            **kwargs,
        )
        return EEGDataset(windows)


def find_device(device=None):
    """Determines the device tat can be used for computation.

    Parameters
    ----------
    device: str
        Name of the device to use.

    Returns
    -------
    str:
        The device that will be used for computation.
    bool:
        Whether GPU compatible with CUDA is available.
    """
    if device is not None:
        assert isinstance(device, str), "device should be a str."
        return torch.device(device), False
    cuda = torch.cuda.is_available()  # check if GPU is available
    if cuda:
        torch.backends.cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device, cuda


def get_subjects(dataset):
    """Originally named get_groups in eeg_augment

    Parameters
    ----------
    dataset: BaseConcatDataset

    Returns
    -------
    np.array
        Array of shape (1, len(dataset)) that maps XXX???

    Note
    ----
    Also works with windows !
    """
    if (hasattr(dataset, "description")
        and hasattr(dataset, "datasets")
            and "subject" in dataset.description):
        return np.hstack([
            [subj] * len(dataset.datasets[rec])
            for rec, subj in enumerate(
                dataset.description['subject'].values)])
    else:
        return np.arange(len(dataset))


def get_labels(dataset):
    """Extract the labels from a dataset.

    Parameters
    ----------
    dataset: Dataset

    Return
    ------
    np.array
        Array of shape (len(dataset),) that contains the label for each sample.
    """
    return np.array([dataset[i][1] for i in range(len(dataset))])


def get_sessions(dataset):
    """Extract sessions from a EEG dataset.

    Parameters
    ----------
    dataset: BaseConcatDataset


    Returns
    -------
    np.array
        Array of shape (1, len(dataset)) that maps XXX???

    Note
    ----
    Also works with windows !
    """
    if (hasattr(dataset, "description")
        and hasattr(dataset, "datasets")
            and "session" in dataset.description):
        return np.hstack([
            [subj] * len(dataset.datasets[rec])
            for rec, subj in enumerate(
                dataset.description['session'].values)])
    else:
        return np.arange(len(dataset))


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0]
    set_random_seeds(seed + worker_id, find_device()[0], cudnn_benchmark=False)


def downsample(dataset, random_state=None):
    """Dowsample a dataset so that all classes are balanced.

    Parameters
    ----------
    dataset: Dataset
        Dataset object that will be downsampled.
    random_state: int | None | RandomGenerator, optional
        Controls the randomness of the training and testing indices produced.

    Returns
    -------
    skorch.dataset:
        Downsampled subset.

    list:
        Downsampled subjects mask. Useful since the Subset object does not
        contains the subject info (contrary to braindecode.WindowsDataset)
    """
    rng = np.random.default_rng(random_state)

    y = get_labels(dataset)
    scarce_class_count = np.bincount(y).min()
    indices = np.array([], dtype=int)
    downsampled_subjects_mask = np.array([], dtype=int)
    subjects_mask = get_subjects(dataset)

    for i in range(len(np.unique(y))):
        y_i = np.where(y == i)[0]  # y_i is a list of indices (misleading name)
        rng.shuffle(y_i)
        indices = np.hstack((indices, y_i[:scarce_class_count]))
        downsampled_subjects_mask = np.hstack(
            (downsampled_subjects_mask,
             subjects_mask[y_i[:scarce_class_count]]))
    return Subset(dataset, indices), downsampled_subjects_mask


class ClassWiseAugmentation(Transform):
    """Subclass from Transform. Allows create handcrafted augmentations that
    apply different augmentation to each class.

    Parameters
    ----------
    aug_per_class: dict
        Dictionary that has classes as keys and augmentations as values.
    """

    def __init__(self, aug_per_class):
        self.aug_per_class = aug_per_class
        super().__init__()

    def __repr__(self):
        return str(self.aug_per_class)

    def forward(self, X, y):
        tr_X = X.clone()
        # Could be changed to the apply identity transformation to classes for
        # which no augmentation is specified
        for c in np.unique(y):
            if c not in self.aug_per_class.keys():
                raise Error(
                    "Unknown class {} found.\n Make sure to parse a class-wise"
                    "augmentation dict that covers all the classes.".format(c))

        for c in self.aug_per_class.keys():
            mask = y == c
            if any(mask):
                tr_X[mask, ...], _ = self.aug_per_class[c](
                    X[mask, ...], y[mask])
        return tr_X, y
