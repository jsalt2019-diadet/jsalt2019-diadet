#!/bin/bash

# this script expects pyannote-audio to be installed like follows, until I make a
# proper pyannote-audio 2.0 release. DO NOT USE pyannote-audio 1.0 FROM PyPI
# $ git clone git@github.com:pyannote/pyannote-audio.git
# $ cd pyannote-audio
# $ git checkout develop   <--- important: one must used the develop branch
# $ pip install .

. ../cmd.sh
. ../path.sh
set -e

# this script is called like "train.sh AMI.SpeakerDiarization.MixHeadset"
PROTOCOL=$1

# use CLSP "free-gpu" command to request a specific GPU
if [ "$(hostname -d)" == "clsp.jhu.edu" ];then
   CUDA_VISIBLE_DEVICES=`free-gpu`
fi




# models are stored here
EXPERIMENT_DIR="exp/vad/${PROTOCOL}"

# corner case for SRI: we use CHiME5 as training set
if [[ "$PROTOCOL" == SRI.* ]]; then

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
pyannote-speech-detection apply --gpu \
  ${MODEL_PT} ${PROTOCOL} ${EXPERIMENT_DIR}/scores

# apply the pipeline and get RTTMs. finally!
pyannote-pipeline apply --subset=development \
  ${EXPERIMENT_DIR}/pipeline/${PROTOCOL}/train/${PROTOCOL}.development/params.yml \
  ${PROTOCOL} ${EXPERIMENT_DIR}/results

declare -A mapping=( ["AMI.SpeakerDiarization.Array1"]="jsalt19_spkdiar_ami_dev_Array1-01" \
                     ["AMI.SpeakerDiarization.Array2"]="jsalt19_spkdiar_ami_dev_Array2-01" \
                     ["AMI.SpeakerDiarization.MixHeadset"]="jsalt19_spkdiar_ami_dev_Mix-Headset" \
                     ["BabyTrain.SpeakerDiarization.All"]="jsalt19_spkdiar_babytrain_dev" \
                     ["CHiME5.SpeakerDiarization.U01"]="jsalt19_spkdiar_chime5_dev_U01" \
                     ["CHiME5.SpeakerDiarization.U06"]="jsalt19_spkdiar_chime5_dev_U06" \
                     ["SRI.SpeakerDiarization.All"]="jsalt19_spkdiar_sri_dev" )
# FIXME. where exactly should I put the VAD?
cp ${EXPERIMENT_DIR}/results/${PROTOCOL}.development.rttm ${mapping[$PROTOCOL]}/vad.pyannote.rttm

pyannote-pipeline apply --subset=test \
  ${EXPERIMENT_DIR}/pipeline/${PROTOCOL}/train/${PROTOCOL}.development/params.yml \
  ${PROTOCOL} ${EXPERIMENT_DIR}/results

declare -A mapping=( ["AMI.SpeakerDiarization.Array1"]="jsalt19_spkdiar_ami_eval_Array1-01" \
                     ["AMI.SpeakerDiarization.Array2"]="jsalt19_spkdiar_ami_eval_Array2-01" \
                     ["AMI.SpeakerDiarization.MixHeadset"]="jsalt19_spkdiar_ami_eval_Mix-Headset" \
                     ["BabyTrain.SpeakerDiarization.All"]="jsalt19_spkdiar_babytrain_eval" \
                     ["CHiME5.SpeakerDiarization.U01"]="jsalt19_spkdiar_chime5_eval_U01" \
                     ["CHiME5.SpeakerDiarization.U06"]="jsalt19_spkdiar_chime5_eval_U06" \
                     ["SRI.SpeakerDiarization.All"]="jsalt19_spkdiar_sri_dev" )
# FIXME. where exactly should I put the VAD?
cp ${EXPERIMENT_DIR}/results/${PROTOCOL}.test.rttm ${mapping[$PROTOCOL]}/vad.pyannote.rttm
