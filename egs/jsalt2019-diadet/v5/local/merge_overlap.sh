#!/bin/bash
# Copyright 2019 JSALT (Diego Castan)  
# Apache 2.0.
#

vadrttm=${1:-vad.rttm}
ovrttm=${2:-ov.rttm}
outputrttm=${3:-vad_overlap.rttm}
loadenv=${4:-true}
#TESTING(FIXME)
envname=${5:-'pyannote'}

if $loadenv ; then
source activate ${envname}
fi

# Covert merging outputs
echo "Merging VAD and OVERLAP"
echo "./local/merge_rttm.py $vadrttm $ovrttm $outputrttm"
./local/merge_rttm.py $vadrttm $ovrttm $outputrttm
