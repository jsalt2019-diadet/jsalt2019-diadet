#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. datapath.sh

if [ $stage -le 1 ];then
    # Prepare the VoxCeleb1 dataset.  The script also downloads a list from
    # http://www.openslr.org/resources/49/voxceleb1_sitw_overlap.txt that
    # contains the speakers that overlap between VoxCeleb1 and our evaluation
    # set SITW.  The script removes these overlapping speakers from VoxCeleb1.
    local/make_voxceleb1.pl $voxceleb1_root 16 data

    # Prepare the dev portion of the VoxCeleb2 dataset.
    local/make_voxceleb2.pl $voxceleb2_root dev 16 data/voxceleb2_train
fi

if [ $stage -le 2 ];then
    # Prepare babytrain
    local/make_babytrain_spkdet.sh $baby_root $babytrain_list_dir ./data
    local/make_babytrain_spkdiar.sh $baby_root $babytrain_list_dir ./data

    # Prepare enhanced data of dev and eval
    echo "running enhancement on each audio may take much time, we link the precomputed audios here."
    local/make_babytrain_spkdet_enhanced.sh $enhanced_eval_root $babytrain_list_dir ./data
    local/make_babytrain_spkdiar_enhanced.sh $enhanced_eval_root $babytrain_list_dir ./data    
fi

if [ $stage -le 3 ];then
    # Prepare chime5
    local/make_chime5_spkdet_jsalt19.sh $chime5_root $chime5_list_dir ./data
    local/make_chime5_spkdiar_jsalt19.sh $chime5_root $chime5_list_dir ./data
    
    # Prepare enhanced data of dev and eval
    echo "running enhancement on each audio may take much time, we link the precomputed audios here."
    local/make_chime5_spkdet_jsalt19_enhanced.sh $enhanced_eval_root $chime5_list_dir ./data
    local/make_chime5_spkdiar_enhanced.sh $enhanced_eval_root $chime5_list_dir ./data
fi


if [ $stage -le 4 ];then
    # Prepare ami
    local/make_ami_spkdet.sh $ami_root $ami_list_dir ./data
    local/make_ami_spkdiar.sh $ami_root $ami_list_dir ./data
    
    # Prepare enhanced data of dev and eval
    echo "running enhancement on each audio may take much time, we link the precomputed audios here."
    local/make_ami_spkdet_enhanced.sh $enhanced_eval_root $ami_list_dir ./data
    local/make_ami_spkdiar_enhanced.sh $enhanced_eval_root $ami_list_dir ./data
fi

if [ $stage -le 5 ];then
    # Prepare SRI
    local/make_sri_spkdet.sh $sri_root $sri_list_dir ./data
    local/make_sri_spkdet_enhanced.sh $enhanced_eval_root $sri_list_dir ./data
    #local/make_sri_spkdiar.sh $sri_root $sri_list_dir ./data
    
    # Prepare enhanced data of dev and eval
    #echo "running enhancement on each audio may take much time, we link the precomputed audios here."
    #local/make_sri_spkdiar_enhanced_dev_eval.sh $enhanced_eval_root $sri_list_dir ./data
fi




