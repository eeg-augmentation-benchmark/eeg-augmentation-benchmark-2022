from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def compute_score(
        df,
        metric=accuracy_score,
        per_fold=True,
        per_class=None,
        set='test'):
    """Compute the score folllowing a given metric.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame that contains the predictions and ground truth values
    metric: function
        Scoring metric to use.
    per_fold: bool
        Whether to compute the score per fold (True) or the score across all
        folds (False)
    per_class: np.array, default=None
        Array that contains classes label. If not None a score perclass will
        be computed for each of these classes.
    set: str, default='test'
        Set used for scoring. ['train', 'valid', 'test']

    Returns
    -------
    pd.DataFrame
        output DataFrame with a new column which contains each fold score.
    """
    sub_df = df.groupby('set').get_group(set).copy()

    if isinstance(per_class, Iterable):

        def label_f1_score(x, label='0'):
            report = classification_report(
                x.y_pred, x.y_true, output_dict=True)
            return float(report[label]['f1-score'])

        df_label_list = []
        for label in per_class:
            new_df = sub_df.copy()
            new_df['f1_score'] = new_df.apply(
                lambda x: label_f1_score(x, label=label),
                axis=1
            )
            new_df['label'] = label
            df_label_list.append(new_df)

        out_df = pd.concat(df_label_list)
        out_df.reset_index(inplace=True)
        return out_df

    if not per_fold:
        y_pred, y_true = [], []
        for id in sub_df.index:
            y_pred = np.hstack((y_pred, sub_df.y_pred[id]))
            y_true = np.hstack((y_true, sub_df.y_true[id]))
        sub_df['avg_score'] = metric(y_true=y_true, y_pred=y_pred)
        return sub_df
    else:
        sub_df['score'] = sub_df.apply(
            lambda x: metric(
                y_true=x.y_true,
                y_pred=x.y_pred
            ),
            axis=1
        )
        return sub_df


def compute_relative_improvement(
        df,
        metric=accuracy_score,
        set='test',
        per_class=None,
        percent=False):
    """Add a column relative improvement to the DataFrame. This column
    contains the relative improvement per fold for each augmentation
    compared to the IdentityTransform (no transformation)

    Parameters:
    -----------
    df: DataFrame
        Output from the cross_val_aug function. The score function must
        have been computed upstream.
    set: str, optional
        The set on witch the relative improvement is computed.
    per_class: list
        List of classes for which the score per class should be computed. Use
        list(map(str, np.arange(5))) to generate this list for a dataset that
        contains 5 classes.
    Returns
    --------
    DataFrame

    Note
    ----
    The graphic can be obtained afterwards with the single line command:
    sns.boxplot(
        x='proportion',
        y='relative_improvement',
        data=test_df,
        hue='augmentation')
    """

    sub_df = df[df['set'] == set].copy()

    if isinstance(per_class, Iterable):
        # make sure that the score has been computed
        if 'f1_score' not in list(sub_df.keys()):
            sub_df = compute_score(
                sub_df, per_class=per_class, set=set, metric=metric)

        score_column_name = 'f1_score'

        df_relative = sub_df.set_index(
            ['fold', 'proportion', 'label']).copy()
    else:
        # make sure that the score has been computed
        if 'score' not in list(sub_df.keys()):
            sub_df = compute_score(
                sub_df, per_class=per_class, set=set, metric=metric)
        score_column_name = 'score'

        df_relative = sub_df.set_index(
            ['fold', 'proportion']).copy()

    scores_ref = df_relative.query(
        'augmentation == "IdentityTransform()"')

    for aug in df_relative.augmentation.unique():
        df_relative.loc[
            df_relative.augmentation == aug,
            score_column_name] /= scores_ref[score_column_name]

    df_relative = df_relative.query(
        'augmentation != "IdentityTransform()"'
    )
    df_relative = df_relative.reset_index()
    df_relative[score_column_name] -= 1
    if percent:
        df_relative[score_column_name] *= 100

    return df_relative
