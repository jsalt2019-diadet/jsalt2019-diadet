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

vaddir_gtdiar=vad_spkdetgtdiar
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

xvector_dir=exp/xvectors/$nnet_name

# create grouth truth diarization datasets
if [ $stage -le 1 ]; then
    # prepared datasets with subsegments based on ground truth diarization clusters
    # each subsegment is defined by a binary vad
    for db in jsalt19_spkdet_{babytrain,ami,sri}{,_enhanced}_{dev,eval}
    do
	name=${db}_test
	rttm=data/${name}/diarization.rttm
	local/make_diar_data.sh --cmd "$train_cmd" --nj 5 --min_dur $min_dur_spkdet_subsegs data/$name $rttm data/${name}_spkdetgtdiar $vaddir_gtdiar
    done

fi


if [ $stage -le 2 ]; then
    # Extracts x-vectors for test with ground truth diarization
    for db in jsalt19_spkdet_{babytrain,ami,sri}{,_enhanced}_{dev,eval}
    do
	name=${db}_test_spkdetgtdiar
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
					     $nnet_dir data/$name \
					     $xvector_dir/$name
    done
fi


if [ $stage -le 3 ]; then
    # combine enroll and test xvectors for step 042
    # with ground truth diarization
    for dset in jsalt19_spkdet_{babytrain,ami,sri}{,_enhanced}
    do
	for part in dev eval
	do
	    db=${dset}_${part}
	    for dur in 5 15 30
	    do
		if [ ! -d data/${db}_enr${dur} ];then
		    continue
		fi
		echo "combining ${db}_enr${dur}_test_spkdetgtdiar"
		mkdir -p $xvector_dir/${db}_enr${dur}_test_spkdetgtdiar
		cat $xvector_dir/${db}_{enr${dur},test_spkdetgtdiar}/xvector.scp \
		    > $xvector_dir/${db}_enr${dur}_test_spkdetgtdiar/xvector.scp
	    done
	done
    done
fi


exit
