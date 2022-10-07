#!/bin/sh

python plots/plot_learning_curve.py outputs/physionet/learning_curves/lr_all_transforms.pkl --tfs-cat frequency --col
python plots/plot_learning_curve.py outputs/physionet/learning_curves/lr_all_transforms.pkl --tfs-cat frequency --col --png
python plots/plot_learning_curve.py outputs/physionet/learning_curves/lr_all_transforms.pkl --tfs-cat time --col
python plots/plot_learning_curve.py outputs/physionet/learning_curves/lr_all_transforms.pkl --tfs-cat time --col --png
python plots/plot_learning_curve.py outputs/physionet/learning_curves/lr_all_transforms.pkl --tfs-cat sensors --col
python plots/plot_learning_curve.py outputs/physionet/learning_curves/lr_all_transforms.pkl --tfs-cat sensors --col --png
python plots/plot_learning_curve.py outputs/physionet/learning_curves/lr_all_transforms.pkl --tfs-cat rotations --col
python plots/plot_learning_curve.py outputs/physionet/learning_curves/lr_all_transforms.pkl --tfs-cat rotations --col --png

python plots/plot_final_comparison.py outputs/physionet/learning_curves/lr_all_transforms.pkl

python plots/plot_learning_curve.py outputs/BCI/learning_curves/lr_all_transforms.pkl --tfs-cat frequency --col
python plots/plot_learning_curve.py outputs/BCI/learning_curves/lr_all_transforms.pkl --tfs-cat frequency --col --png
python plots/plot_learning_curve.py outputs/BCI/learning_curves/lr_all_transforms.pkl --tfs-cat time --col
python plots/plot_learning_curve.py outputs/BCI/learning_curves/lr_all_transforms.pkl --tfs-cat time --col --png
python plots/plot_learning_curve.py outputs/BCI/learning_curves/lr_all_transforms.pkl --tfs-cat sensors --col
python plots/plot_learning_curve.py outputs/BCI/learning_curves/lr_all_transforms.pkl --tfs-cat sensors --col --png
python plots/plot_learning_curve.py outputs/BCI/learning_curves/lr_all_transforms.pkl --tfs-cat rotations --col
python plots/plot_learning_curve.py outputs/BCI/learning_curves/lr_all_transforms.pkl --tfs-cat rotations --col --png

python plots/plot_final_comparison.py outputs/BCI/learning_curves/lr_all_transforms.pkl
