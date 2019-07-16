#!/bin/bash
# Copyright 2019 JSALT (Diego Castan)  
# Apache 2.0.
#

EXP_DIR=${1:-./tutorials/models/overlap_detection/AMI.SpeakerDiarization.MixHeadset.train}
dataset=${2:-AMI.SpeakerDiarization.MixHeadset}
from=${3:-1}
to=${4:-1000}
every=${5:-1}
precision=${6:-0.9}
loadenv=${7:-true}
envname=${8:-'pyannote-audio'}

if $loadenv ; then
    source activate $envname
fi
pyannote-overlap-detection validate --gpu --from=${from} --to=${to} --every=${every} --precision=${precision} --chronological ${EXP_DIR} ${dataset}
