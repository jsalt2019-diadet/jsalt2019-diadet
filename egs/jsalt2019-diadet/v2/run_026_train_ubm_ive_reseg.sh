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

num_components=1024 # the number of UBM components (used for VB resegmentation)
# num_components=128 # the number of UBM components (used for VB resegmentation)
ivector_dim=400 # the dimension of i-vector (used for VB resegmentation)
# ivector_dim=50 # the dimension of i-vector (used for VB resegmentation)

dsets_train=(jsalt19_spkdiar_{babytrain,chime5,ami}_train)

#datasets from array to string list
dsets_train="${dsets_train[@]}"

VB_dir=exp/VB

# Variational Bayes resegmentation using the code from Brno University of Technology
# Please see https://speech.fit.vutbr.cz/software/vb-diarization-eigenvoice-and-hmm-priors 
# for details
if [ $stage -le 1 ]; then

  for name in $dsets_train
    do

    # utils/subset_data_dir.sh data/jsalt19_spkdiar_ami_train 40 data/jsalt19_spkdiar_ami_train_40
    # Train the diagonal UBM.
    mkdir -p $VB_dir || exit 1;
    VB/sid/train_diag_ubm.sh --cmd "$train_cmd --mem 10G" \
    --nj 40 --num-threads 8 --subsample 1 --delta-order 0 --apply-cmn false \
    data/$name \
    $num_components $VB_dir/$name/diag_ubm_$num_components

    done
fi

if [ $stage -le 2 ]; then

  for name in $dsets_train
    do

    # Train the i-vector extractor. The UBM is assumed to be diagonal.
    VB/diarization/train_ivector_extractor_diag.sh \
    --cmd "$train_cmd --mem 10G" \
    --ivector-dim $ivector_dim --num-iters 5 --apply-cmn false \
    --num-threads 1 --num-processes 1 --nj 40 \
    $VB_dir/$name/diag_ubm_$num_components/final.dubm  data/$name \
    $VB_dir/$name/extractor_diag_c${num_components}_i${ivector_dim}

    done
fi
