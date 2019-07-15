#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba, Zili Huang, Fei Wu, Jiamin Xie)
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

xvector_dir=exp/xvectors_diar/$nnet_name

be_babytrain_dir=exp/be_diar/$nnet_name/$be_diar_babytrain_name
be_chime5_dir=exp/be_diar/$nnet_name/$be_diar_chime5_name
be_ami_dir=exp/be_diar/$nnet_name/$be_diar_ami_name
be_sri_dir=$be_chime5_dir

# declare -A be_dirs
# be_dirs[babytrain]=$be_babytrain_dir
# be_dirs[chime5]=$be_chime5_dir
# be_dirs[ami]=$be_ami_dir

score_babytrain_dir=exp/diarization/$nnet_name/$be_diar_babytrain_name
score_chime5_dir=exp/diarization/$nnet_name/$be_diar_chime5_name
score_ami_dir=exp/diarization/$nnet_name/$be_diar_ami_name
score_sri_dir=$score_chime5_dir

#dev datasets
dsets_spkdiar_dev_evad=(jsalt19_spkdiar_babytrain_dev jsalt19_spkdiar_chime5_dev_{U01,U06} jsalt19_spkdiar_ami_dev_{Mix-Headset,Array1-01,Array2-01} jsalt19_spkdiar_sri_dev)
dsets_spkdiar_dev_gtvad=(jsalt19_spkdiar_babytrain_dev_gtvad jsalt19_spkdiar_chime5_dev_{U01,U06}_gtvad jsalt19_spkdiar_ami_dev_{Mix-Headset,Array1-01,Array2-01}_gtvad jsalt19_spkdiar_sri_dev_gtvad)
dsets_spkdet_dev_evad=(jsalt19_spkdet_{babytrain,ami,sri}_dev_test)
dsets_spkdet_dev_gtvad=(jsalt19_spkdet_{babytrain,ami,sri}_dev_test_gtvad)

#eval datasets
dsets_spkdiar_eval_evad=($(echo ${dsets_spkdiar_dev_evad[@]} | sed 's@_dev@_eval@g'))
dsets_spkdiar_eval_gtvad=($(echo ${dsets_spkdiar_dev_gtvad[@]} | sed 's@_dev@_eval@g'))
dsets_spkdet_eval_evad=($(echo ${dsets_spkdet_dev_evad[@]} | sed 's@_dev@_eval@g'))
dsets_spkdet_eval_gtvad=($(echo ${dsets_spkdet_dev_gtvad[@]} | sed 's@_dev@_eval@g'))

dsets_dev=(${dsets_spkdiar_dev_evad[@]} ${dsets_spkdiar_dev_gtvad[@]} ${dsets_spkdet_dev_evad[@]} ${dsets_spkdet_dev_gtvad[@]})
dsets_eval=(${dsets_spkdiar_eval_evad[@]} ${dsets_spkdiar_eval_gtvad[@]} ${dsets_spkdet_eval_evad[@]} ${dsets_spkdet_eval_gtvad[@]})
#dsets_dev=(${dsets_spkdiar_dev_evad[@]} ${dsets_spkdiar_dev_gtvad[@]})
#dsets_eval=(${dsets_spkdiar_eval_evad[@]} ${dsets_spkdiar_eval_gtvad[@]})

dsets_test="${dsets_dev[@]} ${dsets_eval[@]}"

# Perform PLDA scoring for babytrain
if [ $stage -le 1 ]; then

    # Perform PLDA scoring on all pairs of segments for each recording.
    for name in $dsets_test
    do
	if [[ "$name" =~ .*babytrain.* ]];then
	    be_dir_i=$be_babytrain_dir
	    score_dir_i=$score_babytrain_dir
	elif [[ "$name" =~ .*chime5.* ]] || [[ "$name" =~ .*sri.* ]];then
	    be_dir_i=$be_chime5_dir
	    score_dir_i=$score_chime5_dir
	elif [[ "$name" =~ .*ami.* ]];then
	    be_dir_i=$be_ami_dir
	    score_dir_i=$score_ami_dir
	else
	    echo "Dataset is not babytrain, chime5, ami or sri"
	    exit 1
	fi	    

	echo "compute PLDA affinity matrix for $name"
	(
	    mkdir -p $score_dir_i/$name
	    num_spk=$(wc -l $xvector_dir/${name}/spk2utt | cut -d " " -f 1)
	    nj=$(($num_spk < 40 ? $num_spk:40))

	    steps_kaldi_diar/score_plda.sh --cmd "$train_cmd --mem 16G" \
					   --nj $nj $be_dir_i $xvector_dir/$name \
					   $score_dir_i/$name/plda_scores
	) &
    done

fi
wait

#thresholds="-2.0 -1.5 -1.2 -1.1 -1 -0.9 -0.8 -0.7 -0.5 -0.4 -0.3 -0.2 -0.1 -0.05 0 0.05 0.1 0.2 0.3 0.4 0.5"
thresholds="-1.5 -1.2 -1.1 -1 -0.9 -0.8 -0.7 -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1"
num_datasets=${#dsets_dev[@]}

# Cluster the PLDA scores using a stopping threshold
if [ $stage -le 2 ]; then
    
    for((i=0;i<$num_datasets;i++))
    do
	dev_dataset_i=${dsets_dev[$i]}
	eval_dataset_i=${dsets_eval[$i]}
	if [[ "$dev_dataset_i" =~ .*babytrain.* ]];then
	    score_dir_i=$score_babytrain_dir
	elif [[ "$dev_dataset_i" =~ .*chime5.* ]] || [[ "$dev_dataset_i" =~ .*sri.* ]];then
	    score_dir_i=$score_chime5_dir
	elif [[ "$dev_dataset_i" =~ .*ami.* ]];then
	    score_dir_i=$score_ami_dir
	else
	    echo "Dataset is not babytrain, chime5, ami or sri"
	    exit 1
	fi	    
	echo "Tuning clustering threshold for $dev_dataset_i/$eval_dataset_i"
	(
	    best_der=1000
	    best_threshold=0
	    #We loop the threshold on the dev data to get different diarizations
	    for threshold in $thresholds
	    do
		
		num_spk=$(wc -l $score_dir_i/$dev_dataset_i/plda_scores/spk2utt  | cut -d " " -f 1)
		nj=$(($num_spk < 20 ? $num_spk:20))
		steps_kaldi_diar/cluster.sh --cmd "$train_cmd --mem 10G" --nj $nj \
					    --threshold $threshold --rttm-channel 1 $score_dir_i/$dev_dataset_i/plda_scores \
					    $score_dir_i/$dev_dataset_i/plda_scores_t$threshold

		md-eval.pl -r data/$dev_dataset_i/diarization.rttm \
			   -s $score_dir_i/$dev_dataset_i/plda_scores_t$threshold/rttm \
			   2> $score_dir_i/$dev_dataset_i/plda_scores_t$threshold/md-eval.log \
			   > $score_dir_i/$dev_dataset_i/plda_scores_t$threshold/result.md-eval

		num_spk=$(wc -l $score_dir_i/$dev_dataset_i/plda_scores/spk2utt  | cut -d " " -f 1)
		nj=$(($num_spk < 20 ? $num_spk:20))
		steps_kaldi_diar/cluster.sh --cmd "$train_cmd --mem 10G" --nj $nj \
					    --threshold $threshold --rttm-channel 1 $score_dir_i/$eval_dataset_i/plda_scores \
					    $score_dir_i/$eval_dataset_i/plda_scores_t$threshold

		md-eval.pl -r data/$eval_dataset_i/diarization.rttm \
			   -s $score_dir_i/$eval_dataset_i/plda_scores_t$threshold/rttm \
			   2> $score_dir_i/$eval_dataset_i/plda_scores_t$threshold/md-eval.log \
			   > $score_dir_i/$eval_dataset_i/plda_scores_t$threshold/result.md-eval

		der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
			   $score_dir_i/$dev_dataset_i/plda_scores_t$threshold/result.md-eval)
		if [ $(echo $der'<'$best_der | bc -l) -eq 1 ]; then
		    best_der=$der
		    best_threshold=$threshold
		fi
		
	    done
	    echo "$best_threshold" > $score_dir_i/$dev_dataset_i/best_dev_threshold
	    rm -rf $score_dir_i/$dev_dataset_i/plda_scores_tbest
	    rm -rf $score_dir_i/$eval_dataset_i/plda_scores_tbest
	    ln -s plda_scores_t${best_threshold} $score_dir_i/$dev_dataset_i/plda_scores_tbest
	    ln -s plda_scores_t${best_threshold} $score_dir_i/$eval_dataset_i/plda_scores_tbest

	    # eval best with pyannote
	    $train_cmd $score_dir_i/$dev_dataset_i/pyannote.log \
		       local/pyannote_score_diar.sh $dev_dataset_i dev $score_dir_i/$dev_dataset_i/plda_scores_tbest &
	    $train_cmd $score_dir_i/$eval_dataset_i/pyannote.log \
		       local/pyannote_score_diar.sh $eval_dataset_i eval $score_dir_i/$eval_dataset_i/plda_scores_tbest
	    wait
	    
	) &
    done
fi
wait
exit

