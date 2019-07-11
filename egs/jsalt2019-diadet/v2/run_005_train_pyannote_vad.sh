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
                # AMI.SpeakerDiarization.Array1 \
                # AMI.SpeakerDiarization.Array2 \
                # BabyTrain.SpeakerDiarization.All \
                # CHiME5.SpeakerDiarization.U01 \
                # CHiME5.SpeakerDiarization.U06
do
  $pyannote_cmd --gpu 1 exp/vad/${PROTOCOL}/train.log \
      steps_vad/train.sh ${PROTOCOL} &

done

# FIXME: what does this wait do?
wait

# # wait for CHiME5 training to be completed (at least 200 epochs) before submitting tuning on SRI
# FAKE_PROTOCOL_DIR=CHiME5.SpeakerDiarization.U06
# FAKE_EXPERIMENT_DIR="exp/vad/${FAKE_PROTOCOL}"
# MODEL_PT=${FAKE_EXPERIMENT_DIR}/models/train/${FAKE_PROTOCOL}.train/weights/0200.pt
# wait_file ${MODEL_PT}

# PROTOCOL=SRI.SpeakerDiarization.All
# $pyannote_cmd --gpu 1 exp/vad/${PROTOCOL}/train.log steps_vad/train.sh ${PROTOCOL}

