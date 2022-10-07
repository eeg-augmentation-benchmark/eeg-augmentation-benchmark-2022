#!/bin/sh
mkdir -p ./outputs/BCI/logs/

touch ./outputs/BCI/logs/learning_curve.txt
python ./experiments/learning_curve.py \
 -e 1600 \
 -k 1 \
 -n 9 \
 -p 0.5 \
 -r 19 \
 -t IdentityTransform FTSurrogate GaussianNoise SignFlip SmoothTimeMask TimeReverse FrequencyShift ChannelsDropout ChannelsShuffle BandstopFilter ChannelsSymmetry SensorXRotation SensorYRotation SensorZRotation \
 -o ./outputs/BCI/learning_curves \
 -s lr_all_transforms \
 -j 5 \
 -d BCI \
 --proportions 5 \
 --device $1 \
 --cachedir=./tmp \
 | tee ./outputs/BCI/logs/learning_curve.txt