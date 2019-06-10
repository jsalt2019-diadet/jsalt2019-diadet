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

# In this script, we augment SITW dev/eval data with reverberation,
# noise, music, and babble, to use it for test

if [ $stage -le 1 ]; then
    #if we already prepared the noise data for train, we don't have to do it again here
    if [ ! -d "data/chime3background_train" ];then
	kaldi_augmentation/aug_step1_prepare_noise_rir_data.sh data/rirs_info
    fi
fi

if [ $stage -le 2 ]; then
    # Because of the way that wav-reverberate works, we need to apply a correction to SNR to obtain the SNR
    # that we really want, if we mix N files to generate babble (we fix N=7), 
    # SNR_real(dB) = SNR_argument(dB) - 10 log10(N)
    kaldi_augmentation/aug_step2_make_aug_data.sh --mode eval \
						  --snrs "15 10 5 0 -5" \
						  --snrs-babble "24 19 14 9 4" \
						  --num-noises-babble "7" \
						  --normalize-output true \
						  "sitw_dev_test sitw_eval_test"
    #remove all backups
    rm -rf data/sitw_*/.backup

    #change names of directories for babble
    snr_arg=(24 19 14 9 4)
    snr_real=(15 10 5 0 -5)
    for((i=0;i<${#snr_arg[*]};i++))
    do
	mv data/sitw_dev_test_babble_snr${snr_arg[$i]} data/sitw_dev_test_babble_snr${snr_real[$i]}
	mv data/sitw_eval_test_babble_snr${snr_arg[$i]} data/sitw_eval_test_babble_snr${snr_real[$i]}
    done
fi

if [ $stage -le 3 ];then
    #put suffixes to test segments in trial files
    for dset in sitw_dev_test sitw_eval_test
    do
	#for noise
	for noise in noise music babble chime3bg
	do
	    for snr in 15 10 5 0 -5
	    do
		if [ "$noise" == "babble" ];then
		    snr_suff=$(($snr+9))
		else
		    snr_suff=$snr
		fi
		suff="${noise}-snr${snr_suff}"
		name=${dset}_${noise}_snr${snr}
		input_list=data/$dset/trials/core-core.lst
		output_list=data/$name/trials/core-core.lst
		mkdir -p data/$name/trials
		awk -v suff=$suff '{ print $1,$2"-"suff,$3}' $input_list > $output_list
	    done
	done
	#for reverb
	for rt60 in 0.0-0.5 0.5-1.0 1.0-1.5 1.5-4.0
	do
	    suff=reverb-rt60-$rt60
	    name=${dset}_reverb_rt60-$rt60
	    input_list=data/$dset/trials/core-core.lst
	    output_list=data/$name/trials/core-core.lst
	    mkdir -p data/$name/trials
	    awk -v suff=$suff '{ print $1,$2"-"suff,$3}' $input_list > $output_list
	done
    done

fi
