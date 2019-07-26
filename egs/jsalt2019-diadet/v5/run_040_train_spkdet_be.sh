#!/bin/bash
# Copyright       2019   Johns Hopkins University (Author: Jesus Villalba)
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
. datapath.sh 

xvector_dir=exp/xvectors/$nnet_name
be_babytrain_dir=exp/be/$nnet_name/$be_babytrain_name
be_chime5_dir=exp/be/$nnet_name/$be_chime5_name
be_ami_dir=exp/be/$nnet_name/$be_ami_name

#Train back-ends
if [ $stage -le 1 ]; then
    echo "Training back-end for babytrain"
    steps_be/train_be_v1.sh --cmd "$train_cmd" \
    				--lda_dim $lda_dim \
    				--plda_type $plda_type \
    				--y_dim $plda_y_dim --z_dim $plda_z_dim \
    				--w_B $w_B_babytrain --w_W $w_W_babytrain \
    				$xvector_dir/${plda_data}_15s/xvector.scp \
    				data/${plda_data}_15s \
    				$xvector_dir/jsalt19_spkdet_babytrain_train/xvector.scp \
    				data/jsalt19_spkdet_babytrain_train $be_babytrain_dir &

    echo "Training back-end for babytrain using enhanced speech data"
    steps_be/train_be_v1.sh --cmd "$train_cmd" \
    				--lda_dim $lda_dim \
    				--plda_type $plda_type \
    				--y_dim $plda_y_dim --z_dim $plda_z_dim \
    				--w_B $w_B_babytrain --w_W $w_W_babytrain \
    				$xvector_dir/${plda_data}_15s/xvector.scp \
    				data/${plda_data}_15s \
    				$xvector_dir/jsalt19_spkdet_babytrain_enhanced_train/xvector.scp \
    				data/jsalt19_spkdet_babytrain_enhanced_train ${be_babytrain_dir}_enhanced &
fi

if [ $stage -le 2 ]; then

    echo "Training back-end for ami"
    steps_be/train_be_v1.sh --cmd "$train_cmd" \
    				--lda_dim $lda_dim \
    				--plda_type $plda_type \
    				--y_dim $plda_y_dim --z_dim $plda_z_dim \
    				--w_B $w_B_ami --w_W $w_W_ami \
    				$xvector_dir/${plda_data}_15s/xvector.scp \
    				data/${plda_data}_15s \
    				$xvector_dir/jsalt19_spkdet_ami_train/xvector.scp \
    				data/jsalt19_spkdet_ami_train $be_ami_dir &
    
    echo "Training back-end for ami using enhanced speech data"
    steps_be/train_be_v1.sh --cmd "$train_cmd" \
    				--lda_dim $lda_dim \
    				--plda_type $plda_type \
    				--y_dim $plda_y_dim --z_dim $plda_z_dim \
    				--w_B $w_B_ami --w_W $w_W_ami \
    				$xvector_dir/${plda_data}_15s/xvector.scp \
    				data/${plda_data}_15s \
    				$xvector_dir/jsalt19_spkdet_ami_enhanced_train/xvector.scp \
    				data/jsalt19_spkdet_ami_enhanced_train ${be_ami_dir}_enhanced &

fi

if [ $stage -le 3 ]; then

    echo "Training back-end for sri/chime"
    steps_be/train_be_v1.sh --cmd "$train_cmd" \
    				--lda_dim $lda_dim \
    				--plda_type $plda_type \
    				--y_dim $plda_y_dim --z_dim $plda_z_dim \
    				--w_B $w_B_chime5 --w_W $w_W_chime5 \
    				$xvector_dir/${plda_data}_15s/xvector.scp \
    				data/${plda_data}_15s \
    				$xvector_dir/jsalt19_spkdet_chime5_train/xvector.scp \
    				data/jsalt19_spkdet_chime5_train $be_chime5_dir &
   
    echo "Training back-end for sri/chime using enhanced data"
    steps_be/train_be_v1.sh --cmd "$train_cmd" \
    				--lda_dim $lda_dim \
    				--plda_type $plda_type \
    				--y_dim $plda_y_dim --z_dim $plda_z_dim \
    				--w_B $w_B_chime5 --w_W $w_W_chime5 \
    				$xvector_dir/${plda_data}_15s/xvector.scp \
    				data/${plda_data}_15s \
    				$xvector_dir/jsalt19_spkdet_chime5_enhanced_train/xvector.scp \
    				data/jsalt19_spkdet_chime5_enhanced_train ${be_chime5_dir}_enhanced &
fi


wait


