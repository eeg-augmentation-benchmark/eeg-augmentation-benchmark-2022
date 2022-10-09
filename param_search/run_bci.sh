#!/bin/sh
mkdir -p ./outputs/BCI/logs/ && \

touch ./outputs/BCI/logs/param_search.txt
python ./experiments/param_search.py \
 -e 1600 \
 -k 1 \
 -n 9 \
 -p 0.5 \
 -r 19 \
 -o ./outputs/BCI/param_search \
 -s ps_all_transforms \
 -t IdentityTransform FTSurrogate GaussianNoise SmoothTimeMask FrequencyShift ChannelsDropout ChannelsShuffle BandstopFilter SensorXRotation SensorYRotation SensorZRotation \
 -j 5 \
 -d BCI \
 --device $1 \
 --proportion 0 \
 --cachedir=./tmp \
 | tee ./outputs/BCI/logs/param_search.txt