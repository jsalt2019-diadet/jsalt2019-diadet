#!/bin/bash

# this script expects pyannote-audio to be installed like follows, until I make a
# proper pyannote-audio 2.0 release. DO NOT USE pyannote-audio 1.0 FROM PyPI
# $ git clone git@github.com:pyannote/pyannote-audio.git
# $ cd pyannote-audio
# $ git checkout develop   <--- important: one must used the develop branch
# $ pip install .

. path.sh
. cmd.sh
set -e

conda activate pyannote

# this script is called like "train.sh AMI.SpeakerDiarization.MixHeadset"
PROTOCOL=$1

vaddir_supervad=`pwd`/vad_supervad

# use CLSP "free-gpu" command to request a specific GPU
if [ "$(hostname -d)" == "clsp.jhu.edu" ];then
   export CUDA_VISIBLE_DEVICES=`free-gpu`
fi




# # models are stored here
EXPERIMENT_DIR="exp/vad/${PROTOCOL}"



# corner case for SRI: we use CHiME5 as training set
if [[ "$PROTOCOL" == SRI.* ]];then

  FAKE_PROTOCOL="CHiME5.SpeakerDiarization.U06"
  FAKE_EXPERIMENT_DIR="exp/vad/${FAKE_PROTOCOL}"

  # obtain the best model by reading "epoch" in "params.yml" file
  PARAMS_YML=${FAKE_EXPERIMENT_DIR}/models/train/${FAKE_PROTOCOL}.train/validate/${PROTOCOL}.development/params.yml
  BEST_EPOCH=`grep epoch $PARAMS_YML | sed 's/epoch\: //'`
  printf -v BEST_EPOCH "%04d" $BEST_EPOCH
  MODEL_PT=${FAKE_EXPERIMENT_DIR}/models/train/${FAKE_PROTOCOL}.train/weights/${BEST_EPOCH}.pt

else

  # obtain the best epoch by reading the resulting "params.yml" file
  PARAMS_YML=${EXPERIMENT_DIR}/models/train/${PROTOCOL}.train/validate/${PROTOCOL}.development/params.yml
  BEST_EPOCH=`grep epoch $PARAMS_YML | sed 's/epoch\: //'`
  printf -v BEST_EPOCH "%04d" $BEST_EPOCH
  MODEL_PT=${EXPERIMENT_DIR}/models/train/${PROTOCOL}.train/weights/${BEST_EPOCH}.pt

fi



# extract raw VAD scores (before binarization) into ${EXPERIMENT_DIR}/models/scores
pyannote-speech-detection apply --subset=development --gpu \
  ${MODEL_PT} ${PROTOCOL} ${EXPERIMENT_DIR}/scores

pyannote-speech-detection apply --subset=test --gpu \
  ${MODEL_PT} ${PROTOCOL} ${EXPERIMENT_DIR}/scores


# apply the pipeline and get RTTMs. finally!
pyannote-pipeline apply --subset=development \
  ${EXPERIMENT_DIR}/pipeline/${PROTOCOL}/train/${PROTOCOL}.development/params.yml \
  ${PROTOCOL} ${EXPERIMENT_DIR}/results


pyannote-pipeline apply --subset=test \
${EXPERIMENT_DIR}/pipeline/${PROTOCOL}/train/${PROTOCOL}.development/params.yml \
${PROTOCOL} ${EXPERIMENT_DIR}/results







