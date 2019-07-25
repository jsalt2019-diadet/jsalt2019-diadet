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
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
vaddiardir=`pwd`/vad_diar

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. datapath.sh
. $config_file

# In this script, we augment the Voxceleb data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.

if [ "$nnet_data" == "voxceleb_div2" ] && [ "$plda_data" == "voxceleb_div2" ];then
    echo "According to your configuration file, you don't need to do augmentations"
    exit
fi


if [ $stage -le 1 ]; then
    if [ ! -d "data/chime3background_train" ];then
	if [ "$(hostname -d)" == "clsp.jhu.edu" ];then
	    kaldi_augmentation/aug_step1_prepare_noise_rir_data.sh data/rirs_info
	else
	    kaldi_augmentation/aug_step1_prepare_noise_rir_data.sh \
		--musan_src_location $musan_root --rirs_src_location $rirs_root \
		--DEMAND_src_location $demand_root --chime3background_src_location $chime3_root \
		data/rirs_info
	fi
    fi
fi

if [ $stage -le 2 ]; then
    kaldi_augmentation/aug_step2_make_aug_data.sh --mode train \
						  --snrs-noise "15:10:5:0:13:8" \
						  --snrs-music "15:10:8:5" \
						  --snrs-babble "20:17:15:13:10" \
                                  		  --snrs-chime3bg "15:10:8:5" \
						  --rt60s "0.0:0.5 0.5:1.0" \
						  --normalize-output true \
						  --stage 1 \
						  --make-reverb-plus-noise true \
						  "voxceleb"
    #remove all backups
    rm -rf data/voxceleb*/.backup
fi

if [ $stage -le 3 ];then
    # we merge the agumented datasets
    # merge all datasets with noise but without reverberation
    export TMPDIR=data/tmp
    mkdir -p $TMPDIR
    utils/combine_data.sh --extra-files "utt2info" data/voxceleb_allnoises \
			  data/voxceleb_{noise,music,babble,chime3bg}_snr*-*

    # merge all datasets with noise and reverberation
    utils/combine_data.sh --extra-files "utt2info" data/voxceleb_allreverbs_allnoises \
			  data/voxceleb_reverb_rt60-{0.0-0.5,0.5-1.0} \
			  data/voxceleb_reverb_rt60-{0.0-0.5,0.5-1.0}_{noise,music,babble,chime3bg}_snr*-*
    
    #delete intermediate data
    #rm -rf data/voxceleb_{reverb,music,noise,chime3bg,babble}*
    unset TMPDIR
fi

if [ $stage -le 4 ];then
    #subsample and merge datasets
    combined_str=""
    for name in voxceleb_allnoises voxceleb_allreverbs_allnoises
    do
	combine_str="$combine_str data/${name}_sub"
	utils/subset_data_dir.sh data/${name} $(wc -l data/voxceleb/utt2spk | awk '{ print int($1)}') data/${name}_sub
    done
    utils/combine_data.sh data/voxceleb_aug_sub $combine_str 
fi

exit
