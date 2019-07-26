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

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file


if [ $stage -le 1 ];then
    
    # Make filterbanks for the augmented data, dev eval.  Note that we do not compute a new
    # vad.scp file here.  Instead, we use the vad.scp from the clean version of
    # the list.
    for dset in sitw_dev_test sitw_eval_test
    do
	#for noise
	for noise in noise music babble chime3bg
	do
	    for snr in 15 10 5 0 -5
	    do
		name=${dset}_${noise}_snr${snr}
		steps_pyfe/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/pymfcc_16k.conf --nj 40 --cmd "$train_cmd" \
      				   data/$name exp/make_mfcc $mfccdir
		fix_data_dir.sh data/$name
	    done
	done
	#for reverb
	for rt60 in 0.0-0.5 0.5-1.0 1.0-1.5 1.5-4.0
	do
	    name=${dset}_reverb_rt60-$rt60
	    steps_pyfe/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/pymfcc_16k.conf --nj 40 --cmd "$train_cmd" \
      			       data/$name exp/make_mfcc $mfccdir
	    fix_data_dir.sh data/$name
	done
    done

fi

    
exit
