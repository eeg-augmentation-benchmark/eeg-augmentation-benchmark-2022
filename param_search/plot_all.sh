#!/bin/sh

python plots/plot_param_search.py outputs/physionet/param_search/ps_all_transforms.pkl --tfs-cat frequency
python plots/plot_param_search.py outputs/physionet/param_search/ps_all_transforms.pkl --tfs-cat frequency --png
python plots/plot_param_search.py outputs/physionet/param_search/ps_all_transforms.pkl --tfs-cat time --col
python plots/plot_param_search.py outputs/physionet/param_search/ps_all_transforms.pkl --tfs-cat time --col --png
python plots/plot_param_search.py outputs/physionet/param_search/ps_all_transforms.pkl --tfs-cat sensors --col
python plots/plot_param_search.py outputs/physionet/param_search/ps_all_transforms.pkl --tfs-cat sensors --col --png
python plots/plot_param_search.py outputs/physionet/param_search/ps_all_transforms.pkl --tfs-cat rotations
python plots/plot_param_search.py outputs/physionet/param_search/ps_all_transforms.pkl --tfs-cat rotations --png

python plots/plot_param_search.py outputs/BCI/param_search/ps_all_transforms.pkl --tfs-cat frequency
python plots/plot_param_search.py outputs/BCI/param_search/ps_all_transforms.pkl --tfs-cat frequency --png
python plots/plot_param_search.py outputs/BCI/param_search/ps_all_transforms.pkl --tfs-cat time --col
python plots/plot_param_search.py outputs/BCI/param_search/ps_all_transforms.pkl --tfs-cat time --col --png
python plots/plot_param_search.py outputs/BCI/param_search/ps_all_transforms.pkl --tfs-cat sensors --col
python plots/plot_param_search.py outputs/BCI/param_search/ps_all_transforms.pkl --tfs-cat sensors --col --png
python plots/plot_param_search.py outputs/BCI/param_search/ps_all_transforms.pkl --tfs-cat rotations
python plots/plot_param_search.py outputs/BCI/param_search/ps_all_transforms.pkl --tfs-cat rotations --png
