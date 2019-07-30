#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

#using energy vad and gt vad
xvector_dir=exp/xvectors_diar/$nnet_name
#using lstm vad
#xvector_dir=/export/fs01/jsalt19/lbullock/jsalt2019-diadet/egs/jsalt2019-diadet/v5/exp/xvectors_diar/$nnet_name

#dsets_spkdiar_test_evad=(jsalt19_spkdiar_babytrain_{dev,eval} jsalt19_spkdiar_chime5_{dev,eval}_{U01,U06} jsalt19_spkdiar_ami_{dev,eval}_{Mix-Headset,Array1-01,Array2-01} jsalt19_spkdiar_sri_{dev,eval})
#dsets_spkdiar_test_gtvad=(jsalt19_spkdiar_babytrain_{dev,eval}_gtvad jsalt19_spkdiar_chime5_{dev,eval}_{U01,U06}_gtvad jsalt19_spkdiar_ami_{dev,eval}_{Mix-Headset,Array1-01,Array2-01}_gtvad jsalt19_spkdiar_sri_{dev,eval}_gtvad)
#dsets_spkdiar_test_evad=(jsalt19_spkdiar_babytrain_{dev,eval} jsalt19_spkdiar_chime5_{dev,eval}_{U01,U06} jsalt19_spkdiar_ami_{dev,eval}_{Mix-Headset,Array1-01,Array2-01})
#dsets_spkdiar_test_gtvad=(jsalt19_spkdiar_babytrain_{dev,eval}_gtvad jsalt19_spkdiar_chime5_{dev,eval}_{U01,U06}_gtvad jsalt19_spkdiar_ami_{dev,eval}_{Mix-Headset,Array1-01,Array2-01}_gtvad)

#dsets_spkdiar_test_evad=(jsalt19_spkdiar_babytrain_{dev,eval})
#dsets_spkdiar_test_gtvad=(jsalt19_spkdiar_babytrain_{dev,eval}_gtvad)
dsets_spkdiar_test_evad=(jsalt19_spkdiar_chime5_{dev,eval}_{U01,U06})
#dsets_spkdiar_test_gtvad=(jsalt19_spkdiar_ami_{dev,eval}_{Mix-Headset,Array1-01,Array2-01}_gtvad)


#dsets_spkdet_test_evad=(jsalt19_spkdet_babytrain_{dev,eval}_test jsalt19_spkdet_ami_{dev,eval}_test jsalt19_spkdet_sri_{dev,eval}_test)
#dsets_spkdet_test_gtvad=(jsalt19_spkdet_babytrain_{dev,eval}_test_gtvad jsalt19_spkdet_ami_{dev,eval}_test_gtvad jsalt19_spkdet_sri_{dev,eval}_test_gtvad)

#datasets from array to string list
#dsets_test="${dsets_spkdiar_test_evad[@]} ${dsets_spkdiar_test_gtvad[@]} ${dsets_spkdet_test_evad[@]} ${dsets_spkdet_test_gtvad[@]}" 
#dsets_test="${dsets_spkdiar_test_evad[@]} ${dsets_spkdiar_test_gtvad[@]}"
dsets_test="${dsets_spkdiar_test_evad[@]}"

VB_dir=exp/vb_clustering_lstmvad

#transform x-vector to PLDA trans subspace
if [ $stage -le 1 ];then
#if false; then
    for name in $dsets_test
    do
	
	if [[ "$name" =~ .*babytrain.* ]];then
	    lda=lda120_plda_voxceleb_babytrain
	elif [[ "$name" =~ .*chime5.* ]] || [[ "$name" =~ .*sri.* ]];then
	    lda=lda120_plda_voxceleb_chime5
	elif [[ "$name" =~ .*ami.* ]];then
	    lda=lda120_plda_voxceleb_ami
	else
	    echo "$name not found"
	    exit 1
	fi

	mkdir -p $VB_dir/$nnet_name/$lda/$name

	if [ ! -f exp/be_diar/$nnet_name/$lda/transform.t.mat ]; then
	    #take transform matrix from PLDA
	    ivector-copy-plda --binary=false exp/be_diar/$nnet_name/$lda/plda -| head -n 122 | tail -n 121 | \
		copy-matrix - exp/be_diar/$nnet_name/$lda/transform.t.mat
	fi
	
	#transform x-vector for vb clustering
	echo "x-vector -> global-mean + LDA + LN + PLDA trans "
	#energy, gt vad: $xvector_dir/$name/xvector.scp
	ivector-subtract-global-mean scp:$xvector_dir/$name/xvector.scp ark:- | \
	    transform-vec exp/be_diar/$nnet_name/$lda/transform.mat ark:- ark:- | \
	    ivector-normalize-length ark:- ark:- | \
	    transform-vec exp/be_diar/$nnet_name/$lda/transform.t.mat \
			  ark:-  ark,t:$VB_dir/$nnet_name/$lda/$name/xvector.txt

	cp $xvector_dir/$name/utt2spk $VB_dir/$nnet_name/$lda/$name/
    done
fi

#run VB clustering
if [ $stage -le 2 ];then
    for name in $dsets_test
    do
	if [[ "$name" =~ .*babytrain.* ]];then
	    lda=lda120_plda_voxceleb_babytrain
	elif [[ "$name" =~ .*chime5.* ]] || [[ "$name" =~ .*sri.* ]];then
	    lda=lda120_plda_voxceleb_chime5
	elif [[ "$name" =~ .*ami.* ]];then
	    lda=lda120_plda_voxceleb_ami
	else
	    echo "$name not found"
	    exit 1
	fi

	if [ ! -f exp/be_diar/$nnet_name/$lda/plda.mean.txt ]; then
	    #take mean and psi from PLDA
	    ivector-copy-plda --binary=false exp/be_diar/$nnet_name/$lda/plda - | head -n 1 | cut -d\  -f4-123 \
												  > exp/be_diar/$nnet_name/$lda/plda.mean.txt
	    ivector-copy-plda --binary=false exp/be_diar/$nnet_name/$lda/plda - | tail -n 2 | head -n 1 \
		| cut -d\  -f3-122  > exp/be_diar/$nnet_name/$lda/plda.psi.txt
	fi
	
	output_rttm_dir=$VB_dir/$nnet_name/$lda/$name/rttm
	mkdir -p $output_rttm_dir
	init_labels_file=exp/diarization/$nnet_name/$lda/$name/plda_scores_tbest
	#lstm vad ahc result
	#init_labels_file=/export/fs01/jsalt19/lbullock/jsalt2019-diadet/egs/jsalt2019-diadet/v5/exp/diarization/$nnet_name/$lda/$name/plda_scores_tbest

	VB/diarization/VB_clustering.sh	--max-iters 10 --initialize 1 \
					$VB_dir/$nnet_name/$lda/$name $init_labels_file $output_rttm_dir \
					exp/be_diar/$nnet_name/$lda/plda.mean.txt exp/be_diar/$nnet_name/$lda/plda.psi.txt  || exit 1;
	
	./steps_kaldi_diar/make_rttm.py --rttm-channel 1 exp/diarization/$nnet_name/$lda/$name/plda_scores/segments $output_rttm_dir/labels $output_rttm_dir/rttm 
	
    done
fi
