#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
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

xvector_dir=exp/xvectors_diar/$nnet_name

be_dir=exp/be_diar/$nnet_name/$be_diar_name
be_babytrain_dir=exp/be_diar/$nnet_name/$be_diar_babytrain_name
be_chimer_dir=exp/be_diar/$nnet_name/$be_diar_chime5_name
be_ami_dir=exp/be_diar/$nnet_name/$be_diar_ami_name

#Train LDA
if [ $stage -le 1 ];then

    mkdir -p $be_dir
    # Train a LDA model on Voxceleb,
    echo "Train LDA"
    $train_cmd $be_dir/log/lda.log \
	       ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_diar_dim \
	       "ark:ivector-subtract-global-mean scp:$xvector_dir/${plda_diar_data}_128k/xvector.scp ark:- |" \
	       ark:$xvector_dir/${plda_diar_data}_128k/utt2spk $be_dir/transform.mat || exit 1;

fi

# Train PLDA models
if [ $stage -le 2 ]; then
    # Train a PLDA model on Voxceleb,
    echo "Train PLDA"
    $train_cmd $be_dir/log/plda.log \
	       ivector-compute-plda ark:$xvector_dir/${plda_diar_data}_128k/spk2utt \
	       "ark:ivector-subtract-global-mean \
      scp:$xvector_dir/${plda_diar_data}_128k/xvector.scp ark:- \
      | transform-vec $be_dir/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
	       $be_dir/plda || exit 1;
    
    cp $xvector_dir/${plda_diar_data}_128k/mean.vec $be_dir

fi



#Train LDA for babytrain
if [ $stage -le 3 ];then

    mkdir -p $be_babytrain_dir
    # Train a LDA model on Voxceleb+babytrain_train,
    echo "Train LDA for babytrain"
    #Center each dataset with its mean
    ivector-subtract-global-mean scp:$xvector_dir/${plda_diar_data}_128k/xvector.scp ark,t:- > $be_babytrain_dir/xvector.ark.tmp
    ivector-subtract-global-mean scp:$xvector_dir/jsalt19_spkdiar_babytrain_train_gtvad/xvector.scp ark,t:- >> $be_babytrain_dir/xvector.ark.tmp
    #remove duplicate key, that appears for some reason
    awk '{ if(!($1 in l)){ print $0; l[$1]=1}}' $be_babytrain_dir/xvector.ark.tmp | copy-vector ark:- ark:$be_babytrain_dir/xvector.ark
    rm $be_babytrain_dir/xvector.ark.tmp
    
    cat $xvector_dir/${plda_diar_data}_128k/utt2spk $xvector_dir/jsalt19_spkdiar_babytrain_train_gtvad/utt2spk | sort -u > $be_babytrain_dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $be_babytrain_dir/utt2spk > $be_babytrain_dir/spk2utt
    $train_cmd $be_dir/log/lda.log \
	       ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_diar_dim \
	       ark:$be_babytrain_dir/xvector.ark \
	       ark:$be_babytrain_dir/utt2spk $be_babytrain_dir/transform.mat || exit 1;

fi

# Train PLDA model for babytrain
if [ $stage -le 4 ]; then
    # Train a PLDA model on Voxceleb+babytrain_train,
    echo "Train PLDA for babytrain"
    $train_cmd $be_dir/log/plda.log \
	       ivector-compute-plda ark:$be_babytrain_dir/spk2utt \
	       "ark:transform-vec $be_babytrain_dir/transform.mat ark:$be_babytrain_dir/xvector.ark ark:- \
      | ivector-normalize-length ark:- ark:- |" \
	       $be_babytrain_dir/plda || exit 1;
    
    cp $xvector_dir/jsalt19_spkdiar_babytrain_train_gtvad/mean.vec $be_babytrain_dir

fi

