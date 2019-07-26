#!/bin/bash
# Copyright 2017-2018  David Snyder
#           2017-2018  Matthew Maciejewski
#           2019 Latané Bullock, Paola Garcia (JSALT 2019) 
#
# Apache 2.0.
#
# This script trains a UBM and i-vector extractor in preparation
# for VB resegmentation in run_027

. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

num_components=1024 # the number of UBM components (used for VB resegmentation)
ivector_dim=400 # the dimension of i-vector (used for VB resegmentation)

dsets_train_evad=(jsalt19_spkdiar_{babytrain,chime5,ami}_train_evad)

dsets_train_gtvad=($(echo ${dsets_train_evad[@]} | sed 's/_evad/_gtvad/g'))
dsets_train_evad=($(echo ${dsets_train_evad[@]} | sed 's/_evad//g'))


#datasets from array to string list
dsets_train="${dsets_train_gtvad[@]} ${dsets_train_evad[@]}"



VB_dir=exp/VB
VB_models_dir=$VB_dir/models

# Variational Bayes resegmentation using the code from Brno University of Technology
# Please see https://speech.fit.vutbr.cz/software/vb-diarization-eigenvoice-and-hmm-priors 
# for details
if [ $stage -le 1 ]; then

  for name in $dsets_train
    do    

    # Train the diagonal UBM.
    # REDFLAG: this script will call a gmm-select binary
    # which may encounter memory issues, depending on your version of Kaldi
    # the issue came up with chime5-train in which files were very long

    mkdir -p $VB_models_dir || exit 1;
    VB/sid/train_diag_ubm.sh --cmd "$train_cmd --mem 10G" \
    --nj 40 --num-threads 8 --subsample 1 --delta-order 0 --apply-cmn false \
    data/$name \
    $num_components $VB_models_dir/$name/diag_ubm_$num_components
  done
fi

wait

if [ $stage -le 2 ]; then

  for name in $dsets_train
    do

    # Train the i-vector extractor. The UBM is assumed to be diagonal.
    VB/diarization/train_ivector_extractor_diag.sh \
    --cmd "$train_cmd --mem 10G" \
    --ivector-dim $ivector_dim --num-iters 5 --apply-cmn false \
    --num-threads 1 --num-processes 1 --nj 40 \
    $VB_models_dir/$name/diag_ubm_$num_components/final.dubm  data/$name \
    $VB_models_dir/$name/extractor_diag_c${num_components}_i${ivector_dim}
  done
fi

wait
