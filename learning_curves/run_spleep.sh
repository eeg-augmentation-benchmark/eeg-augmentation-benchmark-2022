#!/bin/sh
mkdir -p ./outputs/physionet/logs/ && \
touch ./outputs/physionet/logs/learning_curve.txt && \
python ./experiments/learning_curve.py \
 -e 300 \
 -k 10 \
 -n 78 \
 -p 0.5 \
 -r 19 \
 -o ./outputs/physionet/learning_curves \
 -s lr_all_transforms \
 -t IdentityTransform FTSurrogate GaussianNoise SignFlip SmoothTimeMask TimeReverse FrequencyShift ChannelsDropout ChannelsShuffle BandstopFilter ChannelsSymmetry SensorXRotation SensorYRotation SensorZRotation \
 -j 5 \
 -d SleepPhysionet \
 --device $1 \
 --downsampling \
 --cachedir=./tmp \
 | tee ./outputs/physionet/logs/learning_curve.txt