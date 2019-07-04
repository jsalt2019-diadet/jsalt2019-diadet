#!/bin/bash
# Copyright 2017-2018  David Snyder
#           2017-2018  Matthew Maciejewski
#           2019 Latané Bullock, Paola Garcia (JSALT 2019) 
#
# Apache 2.0.
#
# This recipe demonstrates the use of x-vectors for speaker diarization.
# The scripts are based on the recipe in ../v1/run.sh, but clusters x-vectors
# instead of i-vectors.  It is similar to the x-vector-based diarization system
# described in "Diarization is Hard: Some Experiences and Lessons Learned for
# the JHU Team in the Inaugural DIHARD Challenge" by Sell et al.  The main
# difference is that we haven't implemented the VB resegmentation yet.

. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file


be_dir=exp/be_diar/$nnet_name/$be_diar_name
score_dir=exp/diarization/$nnet_name/$be_diar_name


# dsets_spkdiar_dev_evad=(jsalt19_spkdiar_babytrain_dev jsalt19_spkdiar_chime5_dev_{U01,U06} jsalt19_spkdiar_ami_dev_{Mix-Headset,Array1-01,Array2-01} jsalt19_spkdiar_sri_dev)
dsets_spkdiar_dev_evad=(jsalt19_spkdiar_babytrain_dev)
dsets_spkdiar_eval_evad=($(echo ${dsets_spkdiar_dev_evad[@]} | sed 's@_dev@_eval@g'))

dsets_test="${dsets_spkdiar_dev_evad[@]} ${dsets_spkdiar_eval_evad[@]}"
echo $dsets_test

VB_dir=exp/VB

num_components=1024 # the number of UBM components (used for VB resegmentation)
# num_components=128 # the number of UBM components (used for VB resegmentation)
ivector_dim=400 # the dimension of i-vector (used for VB resegmentation)
# ivector_dim=50 # the dimension of i-vector (used for VB resegmentation)


if [ $stage -le 1 ]; then

  for name in $dsets_test
    do

    if [[ "$db" =~ .*_babytrain_.* ]];then
	    trained_dir=jsalt19_spkdiar_babytrain_train
    elif [[ "$db" =~ .*_ami_.* ]];then
        trained_dir=jsalt19_spkdiar_ami_train
    elif [[ "$db" =~ .*_chime5_.* ]];then
        trained_dir=jsalt19_spkdiar_chime5_train
    else
        echo "$db not found"
        exit 1
    fi


    output_rttm_dir=$VB_dir/$name/rttm
    mkdir -p $output_rttm_dir || exit 1;
    cat $score_dir/$name/plda_scores_tbest/rttm > $output_rttm_dir/pre_VB_rttm
    cat $score_dir/$name/plda_scores_tbest/result.md-eval > $output_rttm_dir/pre_result.md-eval
    init_rttm_file=$output_rttm_dir/pre_VB_rttm

    # VB resegmentation. In this script, I use the x-vector result to 
    # initialize the VB system. You can also use i-vector result or random 
    # initize the VB system. The following script uses kaldi_io. 
    # You could use `sh ../../../tools/extras/install_kaldi_io.sh` to install it
    VB/diarization/VB_resegmentation.sh --nj 20 --cmd "$train_cmd --mem 10G" \
      --initialize 1 data/$name $init_rttm_file $VB_dir/$name \
      $VB_dir/$trained_dir/diag_ubm_$num_components/final.dubm $VB_dir/$trained_dir/extractor_diag_c${num_components}_i${ivector_dim}/final.ie || exit 1; 
    done
fi

if [ $stage -le 2 ]; then
  
  for name in $dsets_test
    do
    # change channel from 0 to 1
    awk '{$3 = 1 ; print}' $VB_dir/$name/rttm/VB_rttm > $VB_dir/$name/rttm/VB_rttm_v2
    
    # Compute the DER after VB resegmentation
    echo "starting DER analysis"
    mkdir -p $VB_dir/$name/rttm || exit 1;
    md-eval.pl -1 -r data/$name/diarization.rttm\
      -s $VB_dir/$name/rttm/VB_rttm_v2 2> $VB_dir/$name/log/VB_DER.log \
      > $VB_dir/$name/rttm/results.md-eval
    der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
      $VB_dir/$name/rttm/results.md-eval)
    echo "After VB resegmentation, DER: $der%"
    done

fi
