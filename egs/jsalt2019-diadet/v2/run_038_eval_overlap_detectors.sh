#!/bin/bash
# Copyright      2019   JSALT workshop (Author: Diego Castan)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1


config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

exp_dir=exp/overlap_models
out_dir=${exp_dir}/test_raw
config_overlap=config.yml

# SRI overlap model is trained in CHiME5
# tst_vec=(AMI.SpeakerDiarization.MixHeadset BabyTrain.SpeakerDiarization.All SRI.SpeakerDiarization.All)
tst_vec=(AMI.SpeakerDiarization.MixHeadset)
num_dbs=${#tst_vec[@]}

##### TO DO ####
# We select just the net that we want for development purposes
# The net should be selected based on the validation file (nets are still on a training stage)
ovnet=${exp_dir}/train/AMI.SpeakerDiarization.MixHeadset.train/weights/0422.pt
###############

#Train overlap
if [ $stage -le 1 ];then

    mkdir -p $out_dir
    # Train a overlap detection model based on LSTM and SyncNet features
    echo "Test overlap detector"
    for((i=0;i<$num_dbs;i++))
    do
        db=${tst_vec[$i]}
        ( 
            $train_cmd_gpu $exp_dir/log/test_${i}.log \
	       ./local/test_overlap.sh $ovnet $db ${out_dir} || exit 1;
        ) &
    done

fi

##Validate overlap
#if [ $stage -le 2 ];then
#if $validate ; then
#
#    # Validate the models trained in stage 1
#    # Note that the validation can run at the same time that the training
#    # Ready to visualize it with Tensorboard
#    echo "Validate overlap detector"
#    for((i=0;i<$num_dbs;i++))
#    do
#        db=${trn_vec[$i]}
#        ( 
#            echo "$train_cmd_gpu $exp_dir/log/validate_${i}.log \
#                ./local/validate_overlap.sh ${exp_dir}/train/${db}.train $db $from $to $every || exit 1;"
#        ) &
#    done
#fi
#fi
#
#wait
