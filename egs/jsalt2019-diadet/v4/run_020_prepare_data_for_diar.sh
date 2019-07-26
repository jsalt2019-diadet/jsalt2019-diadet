#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

feats_diar=`pwd -P`/exp/feats_diar
storage_name=jsalt19-v4-diar-$(date +'%m_%d_%H_%M')
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;

#datasets:
# - voxceleb
# - speaker detection test part with energy vad and ground truth vad
# - speaker diarization datasets with energy vad and ground truth vad
dsets_train="voxceleb"
dsets_adapt=(jsalt19_spkdiar_{babytrain,chime5,ami}{,_enhanced}_train_gtvad)
dsets_spkdiar_test_evad=(jsalt19_spkdiar_babytrain{,_enhanced}_{dev,eval} jsalt19_spkdiar_chime5{,_enhanced}_{dev,eval}_{U01,U06} jsalt19_spkdiar_ami{,_enhanced}_{dev,eval}_{Mix-Headset,Array1-01,Array2-01})
dsets_spkdiar_test_gtvad=(jsalt19_spkdiar_babytrain{,_enhanced}_{dev,eval}_gtvad jsalt19_spkdiar_chime5{,_enhanced}_{dev,eval}_{U01,U06}_gtvad jsalt19_spkdiar_ami{,_enhanced}_{dev,eval}_{Mix-Headset,Array1-01,Array2-01}_gtvad)

dsets_spkdet_test_evad=(jsalt19_spkdet_babytrain{,_enhanced}_{dev,eval}_test jsalt19_spkdet_ami{,_enhanced}_{dev,eval}_test jsalt19_spkdet_sri{,_enhanced}_{dev,eval}_test jsalt19_spkdet_{ami,babytrain,sri}_{dev,eval}_test_overlap)
dsets_spkdet_test_gtvad=(jsalt19_spkdet_babytrain{,_enhanced}_{dev,eval}_test_gtvad jsalt19_spkdet_ami{,_enhanced}_{dev,eval}_test_gtvad jsalt19_spkdet_sri{,_enhanced}_{dev,eval}_test_gtvad)

#datasets from array to string list"
dsets_adapt="${dsets_adapt[@]}"
dsets_test_evad="${dsets_spkdiar_test_evad[@]} ${dsets_spkdet_test_evad[@]}"
dsets_test_gtvad="${dsets_spkdiar_test_gtvad[@]} ${dsets_spkdet_test_gtvad[@]}"


echo "Stage 1"
if [ $stage -le 1 ];then

    for name in $dsets_train $dsets_adapt $dsets_test_evad $dsets_test_gtvad
    do
    echo $name
	num_utt=$(wc -l data/$name/utt2spk | cut -d " " -f 1)
	nj=$(($num_utt < 40 ? 2:40))
    	steps_kaldi_diar/prepare_feats.sh --nj $nj --cmd "$train_cmd" --storage_name $storage_name \
    					  data/$name data_diar/${name}_cmn $feats_diar/${name}_cmn
    	cp data/$name/vad.scp data_diar/${name}_cmn/
    	if [ -f data/$name/segments ]; then
	    #we need a segments file to aboid fail in fix_data_dir
    	    cp data/$name/segments data_diar/${name}_cmn/
    	fi
    	utils/fix_data_dir.sh data_diar/${name}_cmn
    done
fi


echo "Stage 2"
if [ $stage -le 2 ];then
    # Create segments to extract x-vectors for ground truth VAD
    for name in $dsets_test_gtvad
    do
    echo $name
	echo "0.01" > data_diar/${name}_cmn/frame_shift
	
	# remove segments file if exists because having segments file, it will produce rttm time marks w.r.t to original audio file
	# by removing segments file, we will obtain time marks with respect to the audio cuts.
	rm -f data_diar/${name}_cmn/segments

	# if we have the vad in rttm format but not segments format, convert to segments format
	if [ -f "data/$name/vad.rttm" ] && [ ! -f "data/$name/vad.segments" ];then
	    local/vad_rttm2segments.sh data/$name/vad.rttm > data/$name/vad.segments
	fi
	#if we already have the ground truth vad in segments format we just copy it and create the segmented dataset
	if [ -f "data/$name/vad.segments" ];then
	    # segments with less than 25ms produce errors, we remove then
	    awk '($4-$3)>=0.025' data/$name/vad.segments > data_diar/${name}_cmn/subsegments
	    #create segmented dataset
	    utils/data/subsegment_data_dir.sh data_diar/${name}_cmn \
					      data_diar/${name}_cmn/subsegments data_diar/${name}_cmn_segmented
	else
	    echo "ground truth vad.rttm or vad.segments not found for dataset $name"
	    exit 1
	fi
    done
fi


echo "Stage 3"
if [ $stage -le 3 ];then
    # Create segments to extract x-vectors for energy VAD
    for name in $dsets_train $dsets_test_evad
    do
    echo $name
	echo "0.01" > data_diar/${name}_cmn/frame_shift
	num_utt=$(wc -l data/$name/utt2spk | cut -d " " -f 1)
	nj=$(($num_utt < 10 ? 1:10))
	# remove segments file if exists because having segments file, it will produce rttm time marks w.r.t to original audio file
	# by removing segments file, we will obtain time marks with respect to the audio cuts.
	rm -f data_diar/${name}_cmn/segments
	#create segmented dataset from binary vad
	steps_kaldi_diar/vad_to_segments.sh --nj $nj --cmd "$train_cmd" \
					    data_diar/${name}_cmn data_diar/${name}_cmn_segmented
    done
fi


echo "Stage 4"
if [ $stage -le 4 ];then
    # Create segments to extract x-vectors for adaptation datsets using ground truth VAD
    # requires diarization.rttm
    for name in $dsets_adapt
    do
    echo $name
	echo "0.01" > data_diar/${name}_cmn/frame_shift
	rm -f data_diar/${name}_cmn/segments
	# we already have the ground truth diarization marks to generate segments for training PLDA
	if [ -f "data/$name/diarization.rttm" ];then
	    #we only use segments with more than 1.5 secs
	    local/subsegment_data_dir_from_diar_rttm.sh --min-dur 1.5 data/$name/diarization.rttm data_diar/${name}_cmn
	    hyp_utils/remove_spk_few_utts.sh --min-num-utts 4 data_diar/${name}_cmn_segmented
	else
	    echo "ground truth diarization.rttm not found for dataset $name"
	    exit 1
	fi
    done
fi
