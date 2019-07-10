#!/bin/bash

# this script expects pyannote-audio to be installed like follows, until I make a
# proper pyannote-audio 2.0 release. DO NOT USE pyannote-audio 1.0 FROM PyPI
# $ git clone https://github.com/pyannote/pyannote-audio.git
# $ cd pyannote-audio
# $ git checkout develop   <--- important: one must used the develop branch
# $ pip install .


. ./cmd.sh
. ./path.sh
set -e

for PROTOCOL in AMI.SpeakerDiarization.MixHeadset \
                AMI.SpeakerDiarization.Array1 \
                AMI.SpeakerDiarization.Array2 \
                BabyTrain.SpeakerDiarization.All \
                CHiME5.SpeakerDiarization.U01 \
                CHiME5.SpeakerDiarization.U06 \
                SRI.SpeakerDiarization.All
do
  $train_cmd --gpu 1 exp/vad/${PROTOCOL}/apply.log \
      steps_vad/apply.sh ${PROTOCOL} &
done
wait
