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
. $config_file

vaddir_diar=./vad_${spkdet_diar_name}_overlap
xvector_dir=exp/xvectors/$nnet_name

# create automatic diarization datasets
echo "Stage 1"
if [ $stage -le 1 ]; then

    # prepared datasets with subsegments based on automatic diarization with energy VAD
    # each subsegment is defined by a binary vad
    for db in jsalt19_spkdet_{ami,babytrain,sri}_{dev,eval}_test_overlap
    do
	name=${db}
	if [[ "$db" =~ .*_babytrain_.* ]];then
	    rttm=$rttm_babytrain_dir/$name/plda_scores_t${diar_thr}/rttm
	elif [[ "$db" =~ .*_ami_.* ]];then
	    rttm=$rttm_ami_dir/$name/plda_scores_t${diar_thr}/rttm
	elif [[ "$db" =~ .*_sri_.* ]];then
	    rttm=$rttm_sri_dir/$name/plda_scores_t${diar_thr}/rttm
	else
	    echo "rttm for $db not found"
	    exit 1
	fi
	local/make_diar_data.sh --cmd "$train_cmd" --nj 5 --min_dur $min_dur_spkdet_subsegs data/$name $rttm data/${name}_${spkdet_diar_name} $vaddir_diar
    done
fi

echo "Stage 2"
if [ $stage -le 2 ]; then
    # Extracts x-vectors for test with automatic diarization and energy VAD
    for db in jsalt19_spkdet_{ami,babytrain,sri}_{dev,eval}_test_overlap
    do
	name=${db}_${spkdet_diar_name}
	echo $name
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
					     $nnet_dir data/$name \
					     $xvector_dir/$name
    done
fi

echo "Stage 3"
if [ $stage -le 3 ]; then
    # combine enroll and test xvectors for step 043
    # with automatic diarization energy vad
    for dset in jsalt19_spkdet_{ami,babytrain,sri}_{dev,eval}
    do
	db=${dset}
	for dur in 5 15 30
	do
	    if [ ! -d data/${db}_enr${dur} ];then
		continue
	    fi
	    echo "combining ${db}_enr${dur}_test_overlap_${spkdet_diar_name}"
	    mkdir -p $xvector_dir/${db}_enr${dur}_test_overlap_${spkdet_diar_name}
	    cat $xvector_dir/${db}_{enr${dur},test_overlap_${spkdet_diar_name}}/xvector.scp \
		> $xvector_dir/${db}_enr${dur}_test_overlap_${spkdet_diar_name}/xvector.scp
	done
    done
fi


