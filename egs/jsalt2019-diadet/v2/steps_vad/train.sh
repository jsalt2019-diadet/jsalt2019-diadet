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

# FIXME: should this go in path.sh?
PYANNOTE_DATABASE_CONFIG=/export/fs01/jsalt19/databases/database.yml

# hardcoded VAD configuration file. one might want
# to make it part of config.sh at some point
read -r -d '' MODEL_CONFIG_YML << EOM
task:
   name: SpeechActivityDetection
   params:
      duration: 2.0
      batch_size: 64
      per_epoch: 1
      parallel: 6

data_augmentation:
   name: AddNoise
   params:
      snr_min: 10
      snr_max: 20
      collection: MUSAN.Collection.BackgroundNoise

feature_extraction:
   name: RawAudio
   params:
      sample_rate: 16000

architecture:
   name: pyannote.audio.models.PyanNet
   params:
      rnn:
         unit: LSTM
         hidden_size: 128
         num_layers: 2
         bidirectional: True
      ff:
         hidden_size: [128, 128]

scheduler:
   name: CyclicScheduler
   params:
      learning_rate: auto
      epochs_per_cycle: 14
EOM

read -r -d '' PIPELINE_PARAMS_YML << EOM
min_duration_off: 0.1
min_duration_on: 0.1
offset: THRESHOLD
onset: THRESHOLD
pad_offset: 0.0
pad_onset: 0.0
EOM

read -r -d '' PIPELINE_CONFIG_YML << EOM
pipeline:
   name: pyannote.audio.pipeline.speech_activity_detection.SpeechActivityDetection
   params:
      scores: SCORES
EOM

# FIXME: this is hard-coded for JHU cluster
# one should probably do that differently
# e.g. set this variable outside of this script
if [ "$(hostname -d)" == "clsp.jhu.edu" ];then
   CUDA_VISIBLE_DEVICES=`free-gpu`
fi

# FIXME. Can I store models here?
EXPERIMENT_DIR="exp/vad/${PROTOCOL}"

# create models directory and configuration file
mkdir -p ${EXPERIMENT_DIR}/models
echo "${MODEL_CONFIG_YML}" > ${EXPERIMENT_DIR}/models/config.yml

# train model for 200 epochs on protocol training set
pyannote-speech-detection train --subset=train --gpu --to=200 ${EXPERIMENT_DIR}/models ${PROTOCOL}

# validate the model every 5 epochs on the development set
pyannote-speech-detection validate --subset=development --gpu --chronological --every=5 --to=200  ${EXPERIMENT_DIR}/models/train/${PROTOCOL}.train ${PROTOCOL}

# obtain the best epoch and threshold by reading the resulting "params.yml" file
PARAMS_YML=${EXPERIMENT_DIR}/models/train/${PROTOCOL}.train/validate/${PROTOCOL}.development/params.yml
BEST_EPOCH=`grep epoch $PARAMS_YML | sed 's/epoch\: //'`
printf -v BEST_EPOCH "%04d" $BEST_EPOCH
THRESHOLD=`grep "  onset" $PARAMS_YML | sed 's/  onset\: //'`

# create VAD pipeline directory and configuration file
# ideally, the pipeline should be optimized but to make things faster
# we use the threshold found during the validation step
mkdir -p ${EXPERIMENT_DIR}/pipeline/${PROTOCOL}/train/${PROTOCOL}.development
echo "${PIPELINE_CONFIG_YML}" | sed "s%SCORES%${EXPERIMENT_DIR}\/scores%" > ${EXPERIMENT_DIR}/pipeline/${PROTOCOL}/config.yml
echo "${PIPELINE_PARAMS_YML}" | sed "s/THRESHOLD/${THRESHOLD}/" > ${EXPERIMENT_DIR}/pipeline/${PROTOCOL}/train/${PROTOCOL}.development/params.yml
