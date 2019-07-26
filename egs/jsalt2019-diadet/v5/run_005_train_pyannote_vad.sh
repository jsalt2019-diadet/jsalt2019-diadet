#!/bin/bash

# this script expects pyannote-audio to be installed like follows, until I make a
# proper pyannote-audio 2.0 release. DO NOT USE pyannote-audio 1.0 FROM PyPI
# $ git clone https://github.com/pyannote/pyannote-audio.git
# $ cd pyannote-audio
# $ git checkout develop   <--- important: one must used the develop branch
# $ pip install .
#
# REDFLAG: the pyannote scripts stdout don't go to terminal when 
# running this script, you'll have to go to the log files 
# to see what's happening

. ./cmd.sh
. ./path.sh
set -e


for PROTOCOL in AMI.SpeakerDiarization.MixHeadset \
                AMI.SpeakerDiarization.Array1 \
                AMI.SpeakerDiarization.Array2 \
                BabyTrain.SpeakerDiarization.All \
                CHiME5.SpeakerDiarization.U01 
do

  $pyannote_cmd --gpu 1 exp/vad/${PROTOCOL}/train.log \
      steps_vad/train.sh ${PROTOCOL} &

  # hack in order to avoid a CUDA visible devices error,
  # if we aren't careful they will all request a gpu at almost
  # the exact same time and get assigned the same gpu
  sleep 15

done


# this ensures that CHiME5.SpeakerDiarization.U06 is complete 
# before starting on SRI because the SRI depends on CHIME5
PROTOCOL=CHiME5.SpeakerDiarization.U06
$pyannote_cmd --gpu 1 exp/vad/${PROTOCOL}/train.log \
    steps_vad/train.sh ${PROTOCOL} 


PROTOCOL=SRI.SpeakerDiarization.All
$pyannote_cmd --gpu 1 exp/vad/${PROTOCOL}/train.log \
   steps_vad/train.sh ${PROTOCOL}


# FIXME: what does this wait do?
wait