#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

feats_diar=`pwd -P`/exp/feats_diar
storage_name=jsalt19-v1-diar-$(date +'%m_%d_%H_%M')
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;

#datasets:
# - voxceleb
# - speaker detection test part with energy vad and ground truth vad
# - speaker diarization datasets with energy vad and ground truth vad
datasets="voxceleb \
		    jsalt19_spkdet_babytrain_dev_test jsalt19_spkdet_babytrain_dev_test_gtvad \
		    jsalt19_spkdet_babytrain_eval_test jsalt19_spkdet_babytrain_eval_test_gtvad \
		    jsalt19_spkdiar_babytrain_train_gtvad \
		    jsalt19_spkdiar_babytrain_dev jsalt19_spkdiar_babytrain_dev_gtvad \
		    jsalt19_spkdiar_babytrain_eval jsalt19_spkdiar_babytrain_eval_gtvad"
		

if [ $stage -le 1 ];then

    for name in $datasets    
    do
    	steps_kaldi_diar/prepare_feats.sh --nj 40 --cmd "$train_cmd" --storage_name $storage_name \
    					  data/$name data_diar/${name}_cmn $feats_diar/${name}_cmn
    	cp data/$name/vad.scp data_diar/${name}_cmn/
    	# if [ -f data/$name/segments ]; then
    	#     cp data/$name/segments data_diar/${name}_cmn/
    	# fi
    	utils/fix_data_dir.sh data_diar/${name}_cmn
    done
fi

datasets_evad="voxceleb \
		    jsalt19_spkdet_babytrain_dev_test \
		    jsalt19_spkdet_babytrain_eval_test \
		    jsalt19_spkdiar_babytrain_dev \
		    jsalt19_spkdiar_babytrain_eval"

datasets_gtvad="jsalt19_spkdet_babytrain_dev_test_gtvad \
		    jsalt19_spkdet_babytrain_eval_test_gtvad \
		    jsalt19_spkdiar_babytrain_train_gtvad \
		    jsalt19_spkdiar_babytrain_dev_gtvad \
		    jsalt19_spkdiar_babytrain_eval_gtvad"
		


if [ $stage -le 2 ];then
    # Create segments to extract x-vectors for ground truth VAD
    for name in $datasets_gtvad
    do
	echo "0.01" > data_diar/${name}_cmn/frame_shift
	rm -f data_diar/${name}_cmn/segments
	# if we have the vad in rttm format but not segments format, convert to segments format
	if [ -f "data/$name/vad.rttm" ] && [ ! -f "data/$name/vad.segments" ];then
	    local/vad_rttm2segments.sh data/$name/vad.rttm > data/$name/vad.segments
	fi
	#if we already have the ground truth vad in segments format we just copy it and create the segmented dataset
	if [ -f "data/$name/vad.segments" ];then
	    awk '($4-$3)>=0.025' data/$name/vad.segments > data_diar/${name}_cmn/subsegments
	    #create segmented dataset
	    utils/data/subsegment_data_dir.sh data_diar/${name}_cmn \
					      data_diar/${name}_cmn/subsegments data_diar/${name}_cmn_segmented
	else
	    echo "ground truth vad.rttm or vad.segments not found for dataset $name"
	    exit 1
	    #create segmented dataset from binary vad
	    steps_kaldi_diar/vad_to_segments.sh --nj 10 --cmd "$train_cmd" \
						data_diar/${name}_cmn data_diar/${name}_cmn_segmented
	fi
    done
    exit
fi

if [ $stage -le 3 ];then
    # Create segments to extract x-vectors for energy VAD
    for name in $datasets_evad
    do
	echo "0.01" > data_diar/${name}_cmn/frame_shift
	rm -f data_diar/${name}_cmn/segments
	#create segmented dataset from binary vad
	steps_kaldi_diar/vad_to_segments.sh --nj 10 --cmd "$train_cmd" \
					    data_diar/${name}_cmn data_diar/${name}_cmn_segmented
    done
fi

