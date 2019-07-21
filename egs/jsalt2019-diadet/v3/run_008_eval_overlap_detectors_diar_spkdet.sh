#!/bin/bash
# Copyright      2019   JSALT workshop (Author: Diego Castan)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=2


config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

exp_dir=exp/overlap_models
out_dir=${exp_dir}/test_raw
out_dir_dev=${exp_dir}/test_raw_dev
config_overlap=config.yml
vaddir_ov=`pwd`/vad_ov  # VAD without OV regions

# SRI overlap model is trained in CHiME5
# tst_vec=(AMI.SpeakerDiarization.MixHeadset BabyTrain.SpeakerDiarization.All SRI.SpeakerDiarization.All)
tst_vec=(AMI.SpeakerDiarization.MixHeadset)
val_vec=(AMI.SpeakerDiarization.MixHeadset.development)
num_dbs_tst=${#tst_vec[@]}

#Test overlap in test databases
if [ $stage -le 1 ];then

    mkdir -p ${out_dir}
    mkdir -p ${out_dir_dev}
    # Test overlap detector in eval and dev
    echo "Test overlap detector"
    for((i=0;i<$num_dbs_tst;i++))
    do
        db=${tst_vec[$i]}
        # We select just the net that we want for development purposes
        # The net should be selected based on the validation file (nets are still on a training stage)
        paramfile=${exp_dir}/train/${tst_vec[$i]}.train/validate/${val_vec[$i]}/params.yml
        nit=`cat $paramfile | grep epoch | cut -d' ' -f2 | xargs -I num printf "%04d\n" num`
        ovnet=${exp_dir}/train/${tst_vec[$i]}.train/weights/${nit}.pt

        ( 
            $train_cmd_gpu $exp_dir/log/test_${i}.log \
	       ./local/test_overlap.sh $ovnet $db ${out_dir} || exit 1;
        ) &
    done

fi
wait

# thresholding and conversion
if [ $stage -le 2 ];then

    # Train a overlap detection model based on LSTM and SyncNet features
    echo "Thresholding and converting..."
    for((i=0;i<$num_dbs_tst;i++))
    do
        db=${tst_vec[$i]}
        # We select just the net that we want for development purposes
        # The net should be selected based on the validation file (nets are still on a training stage)
        paramfile=${exp_dir}/train/${tst_vec[$i]}.train/validate/${val_vec[$i]}/params.yml
        offset=`cat $paramfile | grep -w offset: | cut -d' ' -f4`
        onset=`cat $paramfile | grep -w onset: | cut -d' ' -f4`
        #echo $onset
        #echo $offset
	    #echo "./local/thr_and_conv_overlap.sh $db ${out_dir} $onset $offset";
        ( 
            $train_cmd $exp_dir/log/thrcov_${i}.log \
	       ./local/thr_and_conv_overlap.sh $db ${out_dir} $onset $offset || exit 1;
        ) &
    done

fi
wait
exit

# To Kaldi format and RTTM for overlap spkdet
if [ $stage -le 3 ];then

    # Train a overlap detection model based on LSTM and SyncNet features
    echo "Covert to Kaldi for SpkDet and VAD RTTM for SpkDet"
    # for dsetname in babytrain ami sri
    for dsetname in ami
    do
    ovtxt=${out_dir}/overlap_${dsetname}.txt
    ./local/diar2spkdet.py ${ovtxt} ${out_dir}
    dset=jsalt19_spkdet_${dsetname}_eval_test
    cut -d' ' -f1 data/${dset}/segments | fgrep -f - ${out_dir}/overlap.rttm > data/${dset}/overlap.rttm
    cut -d' ' -f1 data/${dset}/segments | fgrep -f - ${out_dir}/segoverlap > data/${dset}/segoverlap
    done

fi

# Remove the overlap areas from the VAD
if [ $stage -le 4 ];then

    # Train a overlap detection model based on LSTM and SyncNet features
    echo "Remove the overlap areas from the VAD"
    # for dsetname in babytrain ami sri
    for dsetname in ami
    do
    dset=jsalt19_spkdet_${dsetname}_eval_test
    ovrttm=data/${dset}/overlap.rttm
    vadrttm=data/${dset}/vad.rttm
    cp -r data/jsalt19_spkdet_${dsetname}_eval_test data/jsalt19_spkdet_${dsetname}_eval_test_overlap
    outputrttm=data/jsalt19_spkdet_${dsetname}_eval_test_overlap/vad_ov.rttm
    ./local/merge_overlap.sh $vadrttm $ovrttm $outputrttm
    hyp_utils/rttm_to_bin_vad.sh --nj 5 $outputrttm data/jsalt19_spkdet_${dsetname}_eval_test_overlap/ $vaddir_ov
    # utils/fix_data_dir.sh data/jsalt19_spkdet_${dsetname}_eval_test_overlap
    done

fi
