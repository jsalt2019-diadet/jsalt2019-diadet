#!/bin/bash
# Copyright 2019 JSALT (Diego Castan)  
# Apache 2.0.
#

EXP_DIR=${1:-./tutorials/models/overlap_detection}
dataset=${2:-AMI.SpeakerDiarization.MixHeadset}
config_ov=${3:-config.yml}
loadenv=${4:-true}
envname=${5:-'pyannote-audio'}

cp $config_ov $EXP_DIR/.
if $loadenv ; then
source activate ${envname}
fi
pyannote-overlap-detection train --gpu --to=1000 ${EXP_DIR} ${dataset}
