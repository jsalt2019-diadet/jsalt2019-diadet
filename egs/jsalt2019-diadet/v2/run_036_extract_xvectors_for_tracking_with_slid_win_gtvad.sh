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
    for db in jsalt19_spkdet_{babytrain,ami,sri}_{dev,eval}
    do
      name=${db}_test_gtvad_cmn_segmented
      namexv=${db}_slid_win_w${window}_s${period}_gtvad
    	steps_kaldi_diar/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
    					     --window ${window} --period ${period} --apply-cmn false \
    					     --min-segment 0.5 $nnet_dir \
    					     data_diar/${name} $xvector_dir/${namexv} &

    done
fi
wait

if [ $stage -le 2 ]; then
    # combine enroll and test xvectors for step 042
    # with ground truth diarization
    for dset in jsalt19_spkdet_{babytrain,ami,sri}
    do
	for part in dev eval
	do
	    db=${dset}_${part}
	    for dur in 5 15 30
	    do
        if [ ! -d $xvector_dir/${db}_enr${dur} ];then
            continue
        fi
		echo "combining ${db}_enr${dur}_test_trackgtdiar"
		mkdir -p $xvector_dir/${db}_enr${dur}_slid_win_w${window}_s${period}_test_track_gtvad
		cat $xvector_dir/${db}_{enr${dur},slid_win_w${window}_s${period}_gtvad}/xvector.scp \
		    > $xvector_dir/${db}_enr${dur}_slid_win_w${window}_s${period}_test_track_gtvad/xvector.scp
	    done
	done
    done
fi

if [ $stage -le 3 ]; then
    # create utt2orig 
    for dsetname in babytrain ami sri
    do
    dset=jsalt19_spkdet_${dsetname}
	for part in dev eval
	do
	    db=${dset}_${part}_slid_win_w${window}_s${period}_gtvad
        echo "Creating utt2orig for ${db}"
        if [ $dsetname = "babytrain" ]; then
            cut -d' ' -f1 $xvector_dir/${db}/segments | awk '{split($1,a,"-"); print $1" "a[1]"-"a[2]"-"a[3]}' > $xvector_dir/${db}/utt2orig
        fi
        if [ $dsetname = "ami" ]; then
            cut -d' ' -f1 $xvector_dir/${db}/segments | awk '{split($1,a,"-"); print $1" "a[1]"-"a[2]"-"a[3]"-"a[4]}' > $xvector_dir/${db}/utt2orig
        fi
        if [ $dsetname = "sri" ]; then
            cut -d' ' -f1 $xvector_dir/${db}/segments | awk '{split($1,a,"-"); print $1" "a[1]"-"a[2]"-"a[3]"-"a[4]"-"a[5]"-"a[6]"-"a[7]}' > $xvector_dir/${db}/utt2orig
        fi
        
	done
    done
fi

exit
