#!/bin/sh
mkdir -p ./outputs/physionet/logs/ && \

touch ./outputs/physionet/logs/param_search.txt
python ./experiments/param_search.py \
 -e 300 \
 -k 10 \
 -n 78 \
 -p 0.5 \
 -r 19 \
 -o ./outputs/physionet/param_search \
 -s ps_all_transforms \
 -t IdentityTransform FTSurrogate GaussianNoise SignFlip SmoothTimeMask TimeReverse FrequencyShift ChannelsDropout ChannelsShuffle BandstopFilter ChannelsSymmetry SensorXRotation SensorYRotation SensorZRotation \
 -j 5 \
 -d SleepPhysionet \
 --device $1 \
 --downsampling \
 --proportion 8 \
 --cachedir=./tmp \
 | tee ./outputs/physionet/logs/param_search.txt
