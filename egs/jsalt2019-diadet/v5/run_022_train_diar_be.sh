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
be_chime5_dir=exp/be_diar/$nnet_name/$be_diar_chime5_name
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



#Train LDA for chime5
if [ $stage -le 5 ];then

    mkdir -p $be_chime5_dir
    # Train a LDA model on Voxceleb+chime5_train,
    echo "Train LDA for chime5"
    #Center each dataset with its mean
    ivector-subtract-global-mean scp:$xvector_dir/${plda_diar_data}_128k/xvector.scp ark,t:- > $be_chime5_dir/xvector.ark.tmp
    ivector-subtract-global-mean scp:$xvector_dir/jsalt19_spkdiar_chime5_train_gtvad/xvector.scp ark,t:- >> $be_chime5_dir/xvector.ark.tmp
    #remove duplicate key, that appears for some reason
    awk '{ if(!($1 in l)){ print $0; l[$1]=1}}' $be_chime5_dir/xvector.ark.tmp | copy-vector ark:- ark:$be_chime5_dir/xvector.ark
    rm $be_chime5_dir/xvector.ark.tmp
    
    cat $xvector_dir/${plda_diar_data}_128k/utt2spk $xvector_dir/jsalt19_spkdiar_chime5_train_gtvad/utt2spk | sort -u > $be_chime5_dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $be_chime5_dir/utt2spk > $be_chime5_dir/spk2utt
    $train_cmd $be_dir/log/lda.log \
	       ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_diar_dim \
	       ark:$be_chime5_dir/xvector.ark \
	       ark:$be_chime5_dir/utt2spk $be_chime5_dir/transform.mat || exit 1;

fi

# Train PLDA model for chime5
if [ $stage -le 6 ]; then
    # Train a PLDA model on Voxceleb+chime5_train,
    echo "Train PLDA for chime5"
    $train_cmd $be_dir/log/plda.log \
	       ivector-compute-plda ark:$be_chime5_dir/spk2utt \
	       "ark:transform-vec $be_chime5_dir/transform.mat ark:$be_chime5_dir/xvector.ark ark:- \
      | ivector-normalize-length ark:- ark:- |" \
	       $be_chime5_dir/plda || exit 1;
    
    cp $xvector_dir/jsalt19_spkdiar_chime5_train_gtvad/mean.vec $be_chime5_dir

fi



#Train LDA for ami
if [ $stage -le 7 ];then

    mkdir -p $be_ami_dir
    # Train a LDA model on Voxceleb+ami_train,
    echo "Train LDA for ami"
    # ind zca whitening
    copy-vector scp:$xvector_dir/jsalt19_spkdiar_ami_train_gtvad/xvector.scp ark:- | local/zca_whitening.py ark:- $be_ami_dir/zca_whitening_ind
    # ood zca whitening
    copy-vector scp:$xvector_dir/${plda_diar_data}_128k/xvector.scp ark:- | local/zca_whitening.py ark:- $be_ami_dir/zca_whitening_ood

    #Center each dataset with its mean
    ivector-subtract-global-mean scp:$xvector_dir/${plda_diar_data}_128k/xvector.scp ark:- | \
	local/zca_whitening.py ark:- $be_ami_dir/zca_whitening_ood $be_ami_dir/zca_whitening_ind ark:- |
	copy-vector ark:- ark,t:- > $be_ami_dir/xvector.ark.tmp
    ivector-subtract-global-mean scp:$xvector_dir/jsalt19_spkdiar_ami_train_gtvad/xvector.scp ark,t:- >> $be_ami_dir/xvector.ark.tmp
    #remove duplicate key, that appears for some reason
    awk '{ if(!($1 in l)){ print $0; l[$1]=1}}' $be_ami_dir/xvector.ark.tmp | copy-vector ark:- ark:$be_ami_dir/xvector.ark
    rm $be_ami_dir/xvector.ark.tmp
    
    cat $xvector_dir/${plda_diar_data}_128k/utt2spk $xvector_dir/jsalt19_spkdiar_ami_train_gtvad/utt2spk | sort -u > $be_ami_dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $be_ami_dir/utt2spk > $be_ami_dir/spk2utt
    $train_cmd $be_dir/log/lda.log \
	       ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_diar_dim \
	       ark:$be_ami_dir/xvector.ark \
	       ark:$be_ami_dir/utt2spk $be_ami_dir/transform.mat || exit 1;

fi

# Train PLDA model for ami
if [ $stage -le 8 ]; then
    # Train a PLDA model on Voxceleb+ami_train,
    echo "Train PLDA for ami"
    $train_cmd $be_dir/log/plda.log \
	       ivector-compute-plda ark:$be_ami_dir/spk2utt \
	       "ark:transform-vec $be_ami_dir/transform.mat ark:$be_ami_dir/xvector.ark ark:- \
      | ivector-normalize-length ark:- ark:- |" \
	       $be_ami_dir/plda || exit 1;
    
    cp $xvector_dir/jsalt19_spkdiar_ami_train_gtvad/mean.vec $be_ami_dir

fi


be_babytrain_dir=exp/be_diar/$nnet_name/${be_diar_babytrain_name}_enhanced
be_chime5_dir=exp/be_diar/$nnet_name/${be_diar_chime5_name}_enhanced
be_ami_dir=exp/be_diar/$nnet_name/${be_diar_ami_name}_enhanced


echo "Train models using enhanced data"
#Train LDA for babytrain
if [ $stage -le 8 ];then

    mkdir -p $be_babytrain_dir
    # Train a LDA model on Voxceleb+babytrain_enhanced_train,
    echo "Train LDA for babytrain using enhanced speech data"
    #Center each dataset with its mean
    ivector-subtract-global-mean scp:$xvector_dir/${plda_diar_data}_128k/xvector.scp ark,t:- > $be_babytrain_dir/xvector.ark.tmp
    ivector-subtract-global-mean scp:$xvector_dir/jsalt19_spkdiar_babytrain_enhanced_train_gtvad/xvector.scp ark,t:- >> $be_babytrain_dir/xvector.ark.tmp
    #remove duplicate key, that appears for some reason
    awk '{ if(!($1 in l)){ print $0; l[$1]=1}}' $be_babytrain_dir/xvector.ark.tmp | copy-vector ark:- ark:$be_babytrain_dir/xvector.ark
    rm $be_babytrain_dir/xvector.ark.tmp
    
    cat $xvector_dir/${plda_diar_data}_128k/utt2spk $xvector_dir/jsalt19_spkdiar_babytrain_enhanced_train_gtvad/utt2spk | sort -u > $be_babytrain_dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $be_babytrain_dir/utt2spk > $be_babytrain_dir/spk2utt
    $train_cmd $be_dir/log/lda.log \
           ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_diar_dim \
           ark:$be_babytrain_dir/xvector.ark \
           ark:$be_babytrain_dir/utt2spk $be_babytrain_dir/transform.mat || exit 1;

fi

# Train PLDA model for babytrain
if [ $stage -le 9 ]; then
    # Train a PLDA model on Voxceleb+babytrain_enhanced_train,
    echo "Train PLDA for babytrain using enhanced speech data"
    $train_cmd $be_dir/log/plda.log \
           ivector-compute-plda ark:$be_babytrain_dir/spk2utt \
           "ark:transform-vec $be_babytrain_dir/transform.mat ark:$be_babytrain_dir/xvector.ark ark:- \
      | ivector-normalize-length ark:- ark:- |" \
           $be_babytrain_dir/plda || exit 1;
    
    cp $xvector_dir/jsalt19_spkdiar_babytrain_enhanced_train_gtvad/mean.vec $be_babytrain_dir

fi



#Train LDA for chime5
if [ $stage -le 10 ];then

    mkdir -p $be_chime5_dir
    # Train a LDA model on Voxceleb+chime5_enhanced_train,
    echo "Train LDA for chime5 using enhanced speech data"
    #Center each dataset with its mean
    ivector-subtract-global-mean scp:$xvector_dir/${plda_diar_data}_128k/xvector.scp ark,t:- > $be_chime5_dir/xvector.ark.tmp
    ivector-subtract-global-mean scp:$xvector_dir/jsalt19_spkdiar_chime5_enhanced_train_gtvad/xvector.scp ark,t:- >> $be_chime5_dir/xvector.ark.tmp
    #remove duplicate key, that appears for some reason
    awk '{ if(!($1 in l)){ print $0; l[$1]=1}}' $be_chime5_dir/xvector.ark.tmp | copy-vector ark:- ark:$be_chime5_dir/xvector.ark
    rm $be_chime5_dir/xvector.ark.tmp
    
    cat $xvector_dir/${plda_diar_data}_128k/utt2spk $xvector_dir/jsalt19_spkdiar_chime5_enhanced_train_gtvad/utt2spk | sort -u > $be_chime5_dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $be_chime5_dir/utt2spk > $be_chime5_dir/spk2utt
    $train_cmd $be_dir/log/lda.log \
           ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_diar_dim \
           ark:$be_chime5_dir/xvector.ark \
           ark:$be_chime5_dir/utt2spk $be_chime5_dir/transform.mat || exit 1;

fi

# Train PLDA model for chime5
if [ $stage -le 11 ]; then
    # Train a PLDA model on Voxceleb+chime5_enhanced_train,
    echo "Train PLDA for chime5 using enhanced speech data"
    $train_cmd $be_dir/log/plda.log \
           ivector-compute-plda ark:$be_chime5_dir/spk2utt \
           "ark:transform-vec $be_chime5_dir/transform.mat ark:$be_chime5_dir/xvector.ark ark:- \
      | ivector-normalize-length ark:- ark:- |" \
           $be_chime5_dir/plda || exit 1;
    
    cp $xvector_dir/jsalt19_spkdiar_chime5_enhanced_train_gtvad/mean.vec $be_chime5_dir

fi



#Train LDA for ami
if [ $stage -le 12 ];then

    mkdir -p $be_ami_dir
    # Train a LDA model on Voxceleb+ami_enhanced_train,
    echo "Train LDA for ami using enhanced speech data"
    # ind zca whitening
    copy-vector scp:$xvector_dir/jsalt19_spkdiar_ami_enhanced_train_gtvad/xvector.scp ark:- | local/zca_whitening.py ark:- $be_ami_dir/zca_whitening_ind
    # ood zca whitening
    copy-vector scp:$xvector_dir/${plda_diar_data}_128k/xvector.scp ark:- | local/zca_whitening.py ark:- $be_ami_dir/zca_whitening_ood

    #Center each dataset with its mean
    ivector-subtract-global-mean scp:$xvector_dir/${plda_diar_data}_128k/xvector.scp ark:- | \
    local/zca_whitening.py ark:- $be_ami_dir/zca_whitening_ood $be_ami_dir/zca_whitening_ind ark:- |
    copy-vector ark:- ark,t:- > $be_ami_dir/xvector.ark.tmp
    ivector-subtract-global-mean scp:$xvector_dir/jsalt19_spkdiar_ami_enhanced_train_gtvad/xvector.scp ark,t:- >> $be_ami_dir/xvector.ark.tmp
    #remove duplicate key, that appears for some reason
    awk '{ if(!($1 in l)){ print $0; l[$1]=1}}' $be_ami_dir/xvector.ark.tmp | copy-vector ark:- ark:$be_ami_dir/xvector.ark
    rm $be_ami_dir/xvector.ark.tmp
    
    cat $xvector_dir/${plda_diar_data}_128k/utt2spk $xvector_dir/jsalt19_spkdiar_ami_enhanced_train_gtvad/utt2spk | sort -u > $be_ami_dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $be_ami_dir/utt2spk > $be_ami_dir/spk2utt
    $train_cmd $be_dir/log/lda.log \
           ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_diar_dim \
           ark:$be_ami_dir/xvector.ark \
           ark:$be_ami_dir/utt2spk $be_ami_dir/transform.mat || exit 1;

fi

# Train PLDA model for ami
if [ $stage -le 13 ]; then
    # Train a PLDA model on Voxceleb+ami_enhanced_train,
    echo "Train PLDA for ami using enhanced speech data"
    $train_cmd $be_dir/log/plda.log \
           ivector-compute-plda ark:$be_ami_dir/spk2utt \
           "ark:transform-vec $be_ami_dir/transform.mat ark:$be_ami_dir/xvector.ark ark:- \
      | ivector-normalize-length ark:- ark:- |" \
           $be_ami_dir/plda || exit 1;
    
    cp $xvector_dir/jsalt19_spkdiar_ami_enhanced_train_gtvad/mean.vec $be_ami_dir

fi



