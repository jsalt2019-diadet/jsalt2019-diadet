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

vaddir_diar=./vad_${track_diar_name}
xvector_dir=exp/xvectors/$nnet_name

# create automatic diarization datasets
if [ $stage -le 1 ]; then

    # prepared datasets with subsegments based on automatic diarization with energy VAD
    # each subsegment is defined by a binary vad
    for db in jsalt19_spkdet_{babytrain,ami}_{dev,eval}
    do
	name=${db}_test
	if [[ "$db" =~ .*_babytrain_.* ]];then
	    rttm=$rttm_babytrain_dir/$name/plda_scores_t${diar_thr}/rttm
	elif [[ "$db" =~ .*_ami_.* ]];then
	    rttm=$rttm_ami_dir/$name/plda_scores_t${diar_thr}/rttm
	else
	    echo "rttm for $db not found"
	    exit 1
	fi
	local/make_diar_data.sh --cmd "$train_cmd" --nj 5 --min_dur $min_dur_track_subsegs data/$name $rttm data/${name}_${track_diar_name} $vaddir_diar
    done

fi

if [ $stage -le 2 ]; then

    # prepared datasets with subsegments based on automatic diarization with energy VAD
    # each subsegment is defined by a binary vad
    for db in jsalt19_spkdet_{babytrain,ami}_{dev,eval}
    do
	name=${db}_test
	name_gt=${db}_test_gtvad
	if [[ "$db" =~ .*_babytrain_.* ]];then
	    rttm=$rttm_babytrain_dir/$name_gt/plda_scores_t${diar_thr}/rttm
	elif [[ "$db" =~ .*_ami_.* ]];then
	    rttm=$rttm_ami_dir/$name_gt/plda_scores_t${diar_thr}/rttm
	else
	    echo "rttm for $db not found"
	    exit 1
	fi
	local/make_diar_data.sh --cmd "$train_cmd" --nj 5 --min_dur $min_dur_track_subsegs data/$name $rttm data/${name}_${track_diar_name}_gtvad ${vaddir_diar}_gtvad
    done

fi


if [ $stage -le 3 ]; then
    # Extracts x-vectors for test with automatic diarization and energy VAD
    for db in jsalt19_spkdet_{babytrain,ami}_{dev,eval}
    do
	name=${db}_test_${track_diar_name}
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
					     $nnet_dir data/$name \
					     $xvector_dir/$name
    done
fi


if [ $stage -le 4 ]; then
    # Extracts x-vectors for test with automatic diarization and ground truth VAD
    for db in jsalt19_spkdet_{babytrain,ami}_{dev,eval}
    do
	name=${db}_test_${track_diar_name}_gtvad
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
					     $nnet_dir data/$name \
					     $xvector_dir/$name
    done
fi



if [ $stage -le 5 ]; then
    # combine enroll and test xvectors for step 043
    # with automatic diarization energy vad
    for dset in jsalt19_spkdet_{babytrain,ami}
    do
	for part in dev eval
	do
	    db=${dset}_${part}
	    for dur in 5 15 30
	    do
		echo "combining ${db}_enr${dur}_test_${track_diar_name}"
		mkdir -p $xvector_dir/${db}_enr${dur}_test_${track_diar_name}
		cat $xvector_dir/${db}_{enr${dur},test_${track_diar_name}}/xvector.scp \
		    > $xvector_dir/${db}_enr${dur}_test_${track_diar_name}/xvector.scp
	    done
	done
    done
fi


if [ $stage -le 6 ]; then
    # combine enroll and test xvectors for step 043
    # with automatic diarization ground truth vad
    for dset in jsalt19_spkdet_{babytrain,ami}
    do
	for part in dev eval
	do
	    db=${dset}_${part}
	    for dur in 5 15 30
	    do
		echo "combining ${db}_enr${dur}_test_${track_diar_name}_gtvad"
		mkdir -p $xvector_dir/${db}_enr${dur}_test_${track_diar_name}_gtvad
		cat $xvector_dir/${db}_{enr${dur},test_${track_diar_name}_gtvad}/xvector.scp \
		    > $xvector_dir/${db}_enr${dur}_test_${track_diar_name}_gtvad/xvector.scp
	    done
	done
    done
fi

