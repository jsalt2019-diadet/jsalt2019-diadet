#!/bin/bash
# Copyright 2019 JSALT (Diego Castan)  
# Apache 2.0.
#

net=${1:-./weights/0960.pt}
dataset=${2:-AMI.SpeakerDiarization.MixHeadset}
dest_dir=${3:-./}
loadenv=${4:-true}
envname=${5:-'pyannote-audio'}

if $loadenv ; then
source activate ${envname}
fi

pyannote-overlap-detection apply --gpu ${net} ${dataset} ${dest_dir}
