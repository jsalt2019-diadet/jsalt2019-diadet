#!/bin/bash
# Copyright      2019   JSALT workshop (Author: Diego Castan)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1

from=700
to=1000
every=10
validate=true

config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

exp_dir=exp/overlap_models
config_overlap=config.yml

# SRI overlap model is trained with CHiME5
trn_vec=(AMI.SpeakerDiarization.MixHeadset BabyTrain.SpeakerDiarization.All CHiME5.SpeakerDiarization.U01)
num_dbs=${#trn_vec[@]}

#Train overlap
if [ $stage -le 1 ];then

    mkdir -p $exp_dir
    # Train a overlap detection model based on LSTM and SyncNet features
    echo "Train overlap detector"
    for((i=0;i<$num_dbs;i++))
    do
        db=${trn_vec[$i]}
        ( 
            $train_cmd_gpu $exp_dir/log/model_${i}.log \
	       ./local/train_overlap.sh $exp_dir $db ${config_overlap} || exit 1;
        ) &
        sleep 15
    done

fi

#Validate overlap
if [ $stage -le 2 ];then
if $validate ; then

    # Validate the models trained in stage 1
    # Note that the validation can run at the same time that the training
    # Ready to visualize it with Tensorboard
    echo "Validate overlap detector"
    for((i=0;i<$num_dbs;i++))
    do
        db=${trn_vec[$i]}
        ( 
            $train_cmd_gpu $exp_dir/log/validate_${i}.log \
                ./local/validate_overlap.sh ${exp_dir}/train/${db}.train $db $from $to $every || exit 1;
        ) &
    done
fi
fi

wait
