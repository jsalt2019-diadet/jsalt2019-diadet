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

num_components=1024 # the number of UBM components (used for VB resegmentation)
ivector_dim=400 # the dimension of i-vector (used for VB resegmentation)



# Variational Bayes resegmentation using the code from Brno University of Technology
# Please see https://speech.fit.vutbr.cz/software/vb-diarization-eigenvoice-and-hmm-priors 
# for details
if [ $stage -le 1 ]; then
  # utils/subset_data_dir.sh data/jsalt19_spkdiar_ami_train 32000 data/jsalt19_spkdiar_ami_train_32k
  # Train the diagonal UBM.
  mkdir -p exp/VB || exit 1;
  kaldi_sid/train_diag_ubm.sh --cmd "$train_cmd --mem 10G" \
    --nj 40 --num-threads 8 --subsample 1 --delta-order 0 --apply-cmn false \
    data/jsalt19_spkdiar_ami_train \
    $num_components exp/VB/diag_ubm_ami_train_$num_components
  
  # Train the i-vector extractor. The UBM is assumed to be diagonal.
  steps_kaldi_diar/train_ivector_extractor_diag.sh \
    --cmd "$train_cmd --mem 10G" \
    --ivector-dim $ivector_dim --num-iters 5 --apply-cmn false \
    --num-threads 1 --num-processes 1 --nj 40 \
    exp/VB/diag_ubm_ami_train_$num_components/final.dubm  data/jsalt19_spkdiar_ami_train \
    exp/VB/extractor_diag_ami_train_c${num_components}_i${ivector_dim}
fi
