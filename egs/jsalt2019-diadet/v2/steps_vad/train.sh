#!/bin/bash

# this script expects pyannote-audio to be installed like follows, until I make a
# proper pyannote-audio 2.0 release. DO NOT USE pyannote-audio 1.0 FROM PyPI
# $ git clone git@github.com:pyannote/pyannote-audio.git
# $ cd pyannote-audio
# $ git checkout develop   <--- important: one must used the develop branch
# $ pip install .
#
# You'll also need to 
# $ pip install pyannote.db.musan

. path.sh
. cmd.sh
set -e

conda activate pyannote

# this script is called like "train.sh AMI.SpeakerDiarization.MixHeadset"
PROTOCOL=$1

TRAIN_EPOCHS=50
VAL_EVERY=2




# use CLSP "free-gpu" command to request a specific GPU
if [ "$(hostname -d)" == "clsp.jhu.edu" ];then
   export CUDA_VISIBLE_DEVICES=`free-gpu`
fi


# models will be stored here
EXPERIMENT_DIR="exp/vad/${PROTOCOL}"

if [[ -d "$EXPERIMENT_DIR/pipeline" ]]; then
  echo "It looks like protocol $PROTOCOL has already been trained"
  echo "Remove $EXPERIMENT_DIR/pipeline and $EXPERIMENT_DIR/modelss if you'd like to re-train. "
  exit 1
fi

# Pyannote will exit if it sees a models dir
if [[ -d "$EXPERIMENT_DIR/models" ]]; then rm -rf $EXPERIMENT_DIR/models; fi


# corner case for SRI: we use CHiME5 as training set
if [[ "$PROTOCOL" == SRI.* ]]; then

  # models from CHiME5 are stored here
  FAKE_PROTOCOL="CHiME5.SpeakerDiarization.U06"
  FAKE_EXPERIMENT_DIR="exp/vad/${FAKE_PROTOCOL}"

  # validate the CHiME5 model every 5 epochs on the development set
  pyannote-speech-detection validate --subset=development \
    --gpu --chronological --parallel=3  --every=$VAL_EVERY --to=$TRAIN_EPOCHS  \
    ${FAKE_EXPERIMENT_DIR}/models/train/${FAKE_PROTOCOL}.train ${PROTOCOL}

  # used to obtain the best threshold by reading the resulting "params.yml" file
  PARAMS_YML=${FAKE_EXPERIMENT_DIR}/models/train/${FAKE_PROTOCOL}.train/validate/${PROTOCOL}.development/params.yml

else

  # create models directory and configuration file
  mkdir -p ${EXPERIMENT_DIR}/models
  cp steps_vad/config/model.config.yml ${EXPERIMENT_DIR}/models/config.yml

  # train model for $TRAIN_EPOCHS epochs on protocol training set
  pyannote-speech-detection train --subset=train \
    --gpu --to=$TRAIN_EPOCHS ${EXPERIMENT_DIR}/models ${PROTOCOL}

  # validate the model every 5 epochs on the development set
  pyannote-speech-detection validate --subset=development\
    --gpu --chronological  --parallel=3 --every=$VAL_EVERY --to=$TRAIN_EPOCHS \
    ${EXPERIMENT_DIR}/models/train/${PROTOCOL}.train ${PROTOCOL}

  # used to obtain the best threshold by reading the resulting "params.yml" file
  PARAMS_YML=${EXPERIMENT_DIR}/models/train/${PROTOCOL}.train/validate/${PROTOCOL}.development/params.yml

fi

# obtain the best threshold by reading the "params.yml" file
THRESHOLD=`grep "  onset" $PARAMS_YML | sed 's/  onset\: //'`

# create VAD pipeline directory and configuration file
# ideally, the pipeline should be optimized but to make things faster
# we use the threshold found during the validation step
mkdir -p ${EXPERIMENT_DIR}/pipeline/${PROTOCOL}/train/${PROTOCOL}.development
cat steps_vad/config/pipeline.config.yml | sed "s%SCORES%${EXPERIMENT_DIR}\/scores%" \
  > ${EXPERIMENT_DIR}/pipeline/${PROTOCOL}/config.yml
cat steps_vad/config/pipeline.params.yml | sed "s/THRESHOLD/${THRESHOLD}/" \
  > ${EXPERIMENT_DIR}/pipeline/${PROTOCOL}/train/${PROTOCOL}.development/params.yml