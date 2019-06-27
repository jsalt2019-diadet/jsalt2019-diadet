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

vaddir_gtdiar=vad_trackgtdiar
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

xvector_dir=exp/xvectors/$nnet_name


if [ $stage -le 1 ]; then
    # Extracts x-vectors for test with sliding window
    for db in jsalt19_spkdet_{babytrain,ami}_{dev,eval}
    do
      name=${db}_test
      namexv=${db}_slid_win
    	steps_kaldi_diar/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
    					     --window 1.5 --period 0.75 --apply-cmn false \
    					     --min-segment 0.5 $nnet_dir \
    					     data/${name} $xvector_dir/${namexv}
    done
fi

if [ $stage -le 2 ]; then
    # combine enroll and test xvectors for step 042
    # with ground truth diarization
    for dset in jsalt19_spkdet_{babytrain,ami}
    do
	for part in dev eval
	do
	    db=${dset}_${part}
	    for dur in 5 15 30
	    do
		echo "combining ${db}_enr${dur}_test_trackgtdiar"
		mkdir -p $xvector_dir/${db}_enr${dur}_slid_win_test_track
		cat $xvector_dir/${db}_{enr${dur},slid_win}/xvector.scp \
		    > $xvector_dir/${db}_enr${dur}_slid_win_test_track/xvector.scp
	    done
	done
    done
fi


exit
