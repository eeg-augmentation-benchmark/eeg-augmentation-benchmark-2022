import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from braindecode.augmentation import IdentityTransform
from braindecode.util import set_random_seeds
from joblib import Parallel
from joblib import delayed
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from skorch.helper import predefined_split
from torch.utils.data.dataset import Subset

from eeg_augmentation_benchmark.utils import get_labels
from eeg_augmentation_benchmark.utils import get_sessions
from eeg_augmentation_benchmark.utils import worker_init_fn


def fit_and_predict(
        clf,
        dataset,
        train_indices,
        test_indices,
        subjects_mask,
        epochs=5,
        proportion=1.,
        fold=None,
        random_state=None,
        unbalanced=False,
        valid_size=0.2):
    """Train a classifier on a train set and use it to infer over a validation
    and test set.

    Parameters
    ----------
    clf: braindecode.EEGClassifier
        Classifier used.
    train_set: Dataset
        Dataset used for the training of the model.
    test_set: Dataset
        Dataset used for inference.
    valid_set: Dataset
        Dataset used for early stopping.
    epochs: int, optional
        Number of epochs for the training
    random_state: int | None | RandomGenerator, optional
        Seed or random number generator to use for the generation of a
        sub-training set.

    Returns
    -------
    list:
        List of dictionaries containing the prediction on the train, test and
        valid sets.
    """
    device = clf.device
    print(f'\nTraining on device: {device}')

    rng = np.random.default_rng(random_state)
    seed = rng.integers(1e5)
    seed_val = rng.integers(1e5)  # use RandomState

    if random_state:
        set_random_seeds(seed, device, cudnn_benchmark=False)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    # Split test & train/validation
    train_valid_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)

    # Split train & validation
    subjects_mask_train_valid = subjects_mask[train_indices]
    train_val_split = GroupShuffleSplit(
        n_splits=1,
        test_size=valid_size,
        random_state=seed_val
    )
    train_indices_new, valid_indices = next(train_val_split.split(
        np.arange(len(train_valid_set)),
        groups=subjects_mask_train_valid
    ))
    train_set = Subset(train_valid_set, train_indices_new)
    valid_set = Subset(train_valid_set, valid_indices)

    clf.set_params(train_split=predefined_split(valid_set))

    # Subsample train set
    labels_mask_train = get_labels(train_set)
    if proportion == 1:
        sub_train_indices = np.arange(len(train_set))
        train_subset = train_set
    else:
        sub_train_indices, _ = train_test_split(
            np.arange(len(train_set)),
            train_size=proportion,
            random_state=seed,
            stratify=labels_mask_train
        )
        train_subset = Subset(train_set, sub_train_indices)

    if unbalanced:
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels_mask_train),
            y=labels_mask_train[sub_train_indices])
        print('Using CrossEntropyLoss with class weights: {}'
              .format(class_weights))
        clf.set_params(criterion__weight=torch.Tensor(
            class_weights).to(device))
        clf.callbacks.append('balanced_accuracy')

    print('Train_size: {}\nValid_size: {}'.format(
        len(sub_train_indices), len(valid_set)
    ))
    clf.fit(train_subset, y=None, epochs=epochs)

    pred_list = []
    for key, subset in zip(
        ['train', 'test', 'valid'],
            [train_subset, test_set, valid_set]):
        y_true = get_labels(subset)
        y_pred = clf.predict(subset)  # no gradient descent.
        prediction = {
            'n_train_windows': len(train_subset),
            'y_pred': y_pred,
            'y_true': y_true,
            'set': key,
            'fold': fold,
            'augmentation': str(clf.iterator_train__transforms),
            'proportion': proportion
        }

        pred_list.append(prediction)
    return pd.DataFrame(pred_list)


def cross_val_aug(
        clf,
        dataset,
        subjects_mask,
        K=5,
        augmentations=[IdentityTransform()],
        n_jobs=1,
        random_state=None,
        **kwargs):
    """Cross-validated fit and predict. Paralellized on augmentations and folds
    """
    indices = np.arange(len(dataset))
    fold_subjects = GroupKFold(n_splits=K)
    rng = np.random.default_rng(random_state)
    fold_random_states = rng.integers(1e5, size=K)

    if clf.device.type == 'cuda':
        clf.set_params(iterator_train__pin_memory=True)
    if n_jobs > 1:
        # Fix for joblib multiprocessing
        clf.set_params(iterator_train__multiprocessing_context='fork')

    fold_scores = Parallel(
        n_jobs=n_jobs)(
        delayed(fit_and_predict)(
            clf=deepcopy(clf).set_params(iterator_train__transforms=aug),
            dataset=dataset,
            train_indices=train_indices,
            test_indices=test_indices,
            fold=k,
            random_state=fold_random_states[k],
            subjects_mask=subjects_mask,
            **kwargs,
        ) for aug in augmentations
            for k, (train_indices, test_indices) in
        enumerate(fold_subjects.split(indices, groups=subjects_mask))
    )

    output = pd.concat(fold_scores, axis=0)
    output.index = np.arange(len(output))
    return output


def fit_and_predict_one_subject(
        dataset,
        clf,
        subject_id,
        random_state_subject=None,
        **kwargs):
    """Train a classifier on a given subject from the dataset

    Parameters
    ----------
    dataset: Dataset
        Dataset that contains multiple subjects.
    clf: EEGClassifier
        Classifier that will be trained and evaluated on one subject.
    subject_id: str
        Key from the dictionary `dataset.split('subject')`.
    random_state_subject: int | None | RandomGenerator, optional
        Seed or random number generator to use for the generation of a
        train/test split.
    **kwargs:
        keyword arguments for `fit_and_predict()`

    Return
    ------
    pd.DataFrame
        The prediction dataframe returned by the fit_and_predict function.
    """
    subject_split = dataset.split('subject')
    print('\nStart training subject n°{}'.format(subject_id))
    subject_set = subject_split[subject_id]
    sessions = get_sessions(subject_set)

    rng = np.random.default_rng(random_state_subject)
    train_session = rng.choice(np.unique(sessions))
    train_indices = np.where(sessions == train_session)[0]
    test_indices = np.where(sessions != train_session)[0]

    kwargs['subjects_mask'] = np.arange(len(dataset))
    return fit_and_predict(
        clf=deepcopy(clf),
        dataset=dataset,
        train_indices=train_indices,
        test_indices=test_indices,
        fold='subject_' + subject_id,
        random_state=rng,
        **kwargs
    )


def parallel_train_subjects(
        dataset,
        clf,
        n_jobs=1,
        random_state=None,
        augmentations=[IdentityTransform()],
        K=1,
        **kwargs):
    """Parallelize `fit_and_predict_one_subject` over subjects.

    Parameters
    ----------
    dataset: Dataset
        Braindecode's BaseConcatDataset or WindowsDataset.
    clf: EEGClassifier
        Classifier that will be copied, trained and evaluated on each subject
    n_jobs: int
        Number of workers that will work in parallel.
    random_state: int | None | RandomGenerator, optional
        Seed or random number generator to use for the generation of the seeds
        used for each subject.
    proportion:
        XXX
    K:
        XXX
    **kwargs:
        keyword arguments for `fit_and_predict()`

    Returns
    -------
    pd.DataFrame
        Dataframe that contains the prediction and ground truth for both
        training and validation samples.
    """

    if clf.device.type == 'cuda':
        clf.set_params(iterator_train__pin_memory=True)
    if n_jobs > 1:
        # Fix for joblib multiprocessing
        clf.set_params(iterator_train__multiprocessing_context='fork')

    subject_split = dataset.split('subject')
    subjects_list = list(subject_split.keys())

    # set a different random seed for each subject
    rng = np.random.default_rng(random_state)
    random_states_sub = rng.integers(1e5, size=len(subjects_list))

    subjects_score = Parallel(n_jobs=n_jobs)(
        delayed(fit_and_predict_one_subject)(
            dataset=dataset,
            clf=deepcopy(clf).set_params(iterator_train__transforms=aug),
            subject_id=subject_id,
            random_state_subject=random_states_sub[i],
            **kwargs,
        ) for aug in augmentations
        for i, subject_id in enumerate(subjects_list)
    )

    output = pd.concat(subjects_score, axis=0)
    output.index = np.arange(len(output))
    return output


def fit_and_score_one_subject_crop(dataset,
                                   clf,
                                   subject_id,
                                   random_state=None,
                                   epochs=5,
                                   proportion=1,
                                   **kwargs):
    """
    subject_id: str
        ID of the subject used for training and prediciton.
    kwargs
        Unused but necessary to make this function compatible with the rest of
        the code.
    XXX
    """

    # Reproductibility
    rng = np.random.default_rng(random_state)
    seed = rng.integers(1e5)
    if random_state:
        set_random_seeds(seed, clf.device, cudnn_benchmark=False)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    # subject split
    subject_split = dataset.split('subject')
    subject_set = subject_split[subject_id]
    print('\nStart training subject n°{}'.format(subject_id))

    # Session split
    session_split = subject_set.split('session')
    train_set = session_split['session_T']
    valid_set = session_split['session_E']

    # Subsample train set
    labels_mask_train = get_labels(train_set)
    if proportion == 1:
        sub_train_indices = np.arange(len(train_set))
        train_subset = train_set
    else:
        sub_train_indices, _ = train_test_split(
            np.arange(len(train_set)),
            train_size=proportion,
            random_state=seed,
            stratify=labels_mask_train
        )
        train_subset = Subset(train_set, sub_train_indices)

    clf.train_split = predefined_split(valid_set)
    clf.fit(train_subset, y=None, epochs=epochs)

    pred_list = []
    for key, subset in zip(
        ['train', 'test', 'valid'],
            [train_subset, valid_set]):
        y_true = get_labels(subset)
        y_pred = clf.predict(subset)  # no gradient descent.
        prediction = {
            'n_train_windows': len(train_subset),
            'y_pred': y_pred,
            'y_true': y_true,
            'set': key,
            'subject': subject_id,
            'augmentation': str(clf.iterator_train__transforms),
        }

        pred_list.append(prediction)
    return pd.DataFrame(pred_list)


def parallel_train_subjects_crop(
        dataset,
        clf,
        n_jobs=1,
        random_state=None,
        augmentations=[IdentityTransform()],
        **kwargs):
    """
    XXX
    """

    if clf.device.type == 'cuda':
        clf.iterator_train__pin_memory = True
    if n_jobs > 1:
        # Fix for joblib multiprocessing
        clf.set_params(iterator_train__multiprocessing_context='fork')
        clf.iterator_train__num_workers = 1
        clf.iterator_train__worker_init_fn = worker_init_fn

    subject_split = dataset.split('subject')
    subjects_list = list(subject_split.keys())

    # set a different random seed for each subject
    rng = np.random.default_rng(random_state)
    random_states_sub = rng.integers(1e5, size=len(subjects_list))

    # train in parallel
    subjects_score = Parallel(n_jobs=n_jobs)(
        delayed(fit_and_score_one_subject_crop)(
            dataset=dataset,
            clf=deepcopy(clf).set_params(iterator_train__transforms=aug),
            subject_id=subject_id,
            random_state=random_states_sub[i],
            **kwargs,
        ) for aug in augmentations
        for i, subject_id in enumerate(subjects_list)
    )

    output = pd.concat(subjects_score, axis=0)
    output.index = np.arange(len(output))
    return output
