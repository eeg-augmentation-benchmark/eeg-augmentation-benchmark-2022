import numpy as np
import torch
from braindecode import EEGClassifier
from braindecode.augmentation import AugmentedDataLoader
from braindecode.augmentation import IdentityTransform
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds
from skorch.callbacks import EarlyStopping
from skorch.callbacks.training import TrainEndCheckpoint

from BAE.utils import find_device
from BAE.utils import worker_init_fn


def get_EEGClassifier(
        model,
        clf_params,
        model_params,
        random_state=None):
    """Generates a braindecode EEGClassifier object. There are many defalut
    parameters defined in this function but they can all be tuned.

    Parameters
    ----------
    model: torch.nn.Sequential
        Torch model to instantiate.
    model_params: dict, optional
        Neral network parameters.
    clf_params: dict, optional
        Classifier parameters for the braindecode.EEGClassifier.

    Returns
    -------
    braindecode.EEGClassifier
    """

    rng = np.random.default_rng(random_state)
    set_random_seeds(
        rng.integers(1e5),
        find_device()[0],
        cudnn_benchmark=False)

    module = model(**model_params)

    if not isinstance(clf_params, dict):
        clf_params = {'module': module}
    else:
        clf_params['module'] = module

    if clf_params['device'].type == 'cuda':
        clf_params['iterator_train__pin_memory'] = True
        # Fix for joblib multiprocessing
        clf_params['iterator_train__multiprocessing_context'] = 'fork'

    return EEGClassifier(**clf_params)


MODEL_PARAMS_BCI = {
    'in_chans': 22,
    'n_classes': 4,
    'input_window_samples': 1125,  # 0.5 sec before cue
    'final_conv_length': 'auto',
}

CLF_PARAMS_BCI = {
    'criterion': torch.nn.NLLLoss,
    'optimizer': torch.optim.AdamW,
    'optimizer__lr': 0.0625 * 0.01,
    'batch_size': 64,  # Robin params
    'device': find_device()[0],
    'callbacks': [
        'accuracy',
        ('early stopping', EarlyStopping(patience=160, load_best=True,)),
        ('final_chackpoint', TrainEndCheckpoint(
            dirname='./outputs/BCI/checkpoints/'))
    ],
    'iterator_train': AugmentedDataLoader,
    'iterator_train__transforms': [IdentityTransform()],
    'iterator_train__num_workers': 4,
    'iterator_train__worker_init_fn': worker_init_fn,
}

MODEL_PARAMS_BCI_CROP = {
    'in_chans': 22,
    'n_classes': 4,
    'input_window_samples': 1125,  # 0.5 sec before cue
    'final_conv_length': 30,
}

CLF_PARAMS_BCI_CROP = {
    'cropped': True,
    'criterion': CroppedLoss,
    'criterion__loss_function': torch.nn.functional.nll_loss,
    'optimizer': torch.optim.AdamW,
    'optimizer__lr': 0.0625 * 0.01,
    'optimizer__weight_decay': 0.,
    'batch_size': 64,  # Robin params
    'device': find_device()[0],
    'callbacks': ['accuracy', ],
    'iterator_train': AugmentedDataLoader,
    'iterator_train__transforms': [IdentityTransform()],
}

MODEL_PARAMS_PHYSIONET = {
    'n_channels': 2,
    'n_classes': 5,
    'sfreq': 100,
    'input_size_s': 30,
    'time_conv_size_s': 0.5,
    'apply_batch_norm': True
}
CLF_PARAMS_PHYSIONET = {
    'batch_size': 16,
    'criterion': torch.nn.CrossEntropyLoss,
    'optimizer': torch.optim.Adam,
    'optimizer__lr': 1e-3,
    'device': find_device()[0],
    'callbacks': [
        'accuracy',
        ('early stopping', EarlyStopping(patience=30, load_best=True)),
        ('final_chackpoint', TrainEndCheckpoint(
            dirname='./outputs/physionet/checkpoints/'))
    ],
    'iterator_train': AugmentedDataLoader,
    'iterator_train__transforms': [IdentityTransform()],
    'iterator_train__shuffle': True,
    'iterator_train__num_workers': 4,
    'iterator_train__worker_init_fn': worker_init_fn,
}
