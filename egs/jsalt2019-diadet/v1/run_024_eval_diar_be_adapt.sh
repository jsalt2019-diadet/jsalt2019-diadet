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
score_babytrain_dir=exp/diarization/$nnet_name/$be_diar_babytrain_name

tst_datasets="jsalt19_spkdet_babytrain_dev_test jsalt19_spkdet_babytrain_dev_test_gtvad \
		    jsalt19_spkdet_babytrain_eval_test jsalt19_spkdet_babytrain_eval_test_gtvad \
		    jsalt19_spkdiar_babytrain_dev jsalt19_spkdiar_babytrain_dev_gtvad \
		    jsalt19_spkdiar_babytrain_eval jsalt19_spkdiar_babytrain_eval_gtvad"

# Perform PLDA scoring for babytrain
if [ $stage -le 1 ]; then

    # Perform PLDA scoring on all pairs of segments for each recording.
    for name in $tst_datasets
    do
	echo "compute PLDA affinity matrix for $name"
	(
	    mkdir -p $score_babytrain_dir/$name
	    #awk '{ print $1,$2}' $xvector_dir/$name/segments > $xvector_dir/$name/utt2spk
	    #utils/utt2spk_to_spk2utt.pl $xvector_dir/$name/utt2spk > $xvector_dir/$name/spk2utt
	    steps_kaldi_diar/score_plda.sh --cmd "$train_cmd --mem 16G" \
					   --nj 20 $be_babytrain_dir $xvector_dir/$name \
					   $score_babytrain_dir/$name/plda_scores
	) &
    done

fi
wait

dev_datasets=(jsalt19_spkdet_babytrain_dev_test jsalt19_spkdet_babytrain_dev_test_gtvad \
						jsalt19_spkdiar_babytrain_dev jsalt19_spkdiar_babytrain_dev_gtvad)
eval_datasets=(jsalt19_spkdet_babytrain_eval_test jsalt19_spkdet_babytrain_eval_test_gtvad \
						 jsalt19_spkdiar_babytrain_eval jsalt19_spkdiar_babytrain_eval_gtvad)

num_datasets=${#dev_datasets[@]}

# Cluster the PLDA scores using a stopping threshold for babytrain.
if [ $stage -le 2 ]; then
    
    for((i=0;i<$num_datasets;i++))
    do
	dev_dataset_i=${dev_datasets[$i]}
	eval_dataset_i=${eval_datasets[$i]}
	echo "Tuning clustering threshold for $dev_dataset_i/$eval_dataset_i"
	(
	    best_der=1000
	    best_threshold=0
	    #We loop the threshold on the dev data to get different diarizations
	    for threshold in -2.0 -1.5 -1.2 -1.1 -1 -0.9 -0.8 -0.7 -0.5 -0.4 -0.3 -0.2 -0.1 -0.05 0 0.05 0.1 0.2 0.3 0.4 0.5
	    do
		steps_kaldi_diar/cluster.sh --cmd "$train_cmd --mem 10G" --nj 20 \
					    --threshold $threshold --rttm-channel 1 $score_babytrain_dir/$dev_dataset_i/plda_scores \
					    $score_babytrain_dir/$dev_dataset_i/plda_scores_t$threshold

		md-eval.pl -r data/$dev_dataset_i/diarization.rttm \
			   -s $score_babytrain_dir/$dev_dataset_i/plda_scores_t$threshold/rttm \
			   2> $score_babytrain_dir/$dev_dataset_i/plda_scores_t$threshold/md-eval.log \
			   > $score_babytrain_dir/$dev_dataset_i/plda_scores_t$threshold/result.md-eval

		steps_kaldi_diar/cluster.sh --cmd "$train_cmd --mem 10G" --nj 20 \
					    --threshold $threshold --rttm-channel 1 $score_babytrain_dir/$eval_dataset_i/plda_scores \
					    $score_babytrain_dir/$eval_dataset_i/plda_scores_t$threshold

		md-eval.pl -r data/$eval_dataset_i/diarization.rttm \
			   -s $score_babytrain_dir/$eval_dataset_i/plda_scores_t$threshold/rttm \
			   2> $score_babytrain_dir/$eval_dataset_i/plda_scores_t$threshold/md-eval.log \
			   > $score_babytrain_dir/$eval_dataset_i/plda_scores_t$threshold/result.md-eval

		der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
			   $score_babytrain_dir/$dev_dataset_i/plda_scores_t$threshold/result.md-eval)
		if [ $(echo $der'<'$best_der | bc -l) -eq 1 ]; then
		    best_der=$der
		    best_threshold=$threshold
		fi
		
	    done
	    echo "$best_threshold" > $score_babytrain_dir/$dev_dataset_i/best_dev_threshold
	    rm -rf $score_babytrain_dir/$dev_dataset_i/plda_scores_tbest
	    rm -rf $score_babytrain_dir/$eval_dataset_i/plda_scores_tbest
	    ln -s plda_scores_t${best_threshold} $score_babytrain_dir/$dev_dataset_i/plda_scores_tbest
	    ln -s plda_scores_t${best_threshold} $score_babytrain_dir/$eval_dataset_i/plda_scores_tbest
	    
	) &
    done
fi
wait
exit

