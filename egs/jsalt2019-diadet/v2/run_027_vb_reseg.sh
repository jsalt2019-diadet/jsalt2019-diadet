#!/bin/bash
# Copyright 2017-2018  David Snyder
#           2017-2018  Matthew Maciejewski
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
ami_dev_Mix="jsalt19_spkdiar_ami_dev_Mix-Headset"


# num_components=1024 # the number of UBM components (used for VB resegmentation)
num_components=128 # the number of UBM components (used for VB resegmentation)
# ivector_dim=400 # the dimension of i-vector (used for VB resegmentation)
ivector_dim=100 # the dimension of i-vector (used for VB resegmentation)


if [ $stage -le 1 ]; then
  output_rttm_dir=exp/VB/rttm
  mkdir -p $output_rttm_dir || exit 1;
  cat exp/diarization//plda_scores_tbest/rttm \ 
    > $output_rttm_dir/pre_VB_rttm
  init_rttm_file=$output_rttm_dir/pre_VB_rttm

  # VB resegmentation. In this script, I use the x-vector result to 
  # initialize the VB system. You can also use i-vector result or random 
  # initize the VB system. The following script uses kaldi_io. 
  # You could use `sh ../../../tools/extras/install_kaldi_io.sh` to install it
  VB/diarization/VB_resegmentation.sh --nj 20 --cmd "$train_cmd --mem 10G" \
    --initialize 1 data/$ami_dev_Mix $init_rttm_file exp/VB \
    exp/VB/diag_ubm_ami_train_$num_components/final.dubm exp/extractor_diag_ami_train_c${num_components}_i${ivector_dim}/final.ie || exit 1; 

  # Compute the DER after VB resegmentation
  mkdir -p exp/VB/results || exit 1;
  md-eval.pl -1 -c 0.25 -r data/$nnet_name/$be_diar_name/$ami_dev_Mix/diarization.rttm \
    -s $output_rttm_dir/VB_rttm 2> exp/VB/log/VB_DER.log \
    > exp/VB/results/VB_DER.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    exp/VB/results/VB_DER.txt)
  echo "After VB resegmentation, DER: $der%"
fi
