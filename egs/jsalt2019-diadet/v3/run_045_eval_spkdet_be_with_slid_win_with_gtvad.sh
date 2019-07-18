#!/bin/bash
# Copyright       2019   Johns Hopkins University (Author: Jesus Villalba)
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

. datapath.sh

snorm=false

xvector_dir=exp/xvectors/$nnet_name
be_babytrain_dir=exp/be/$nnet_name/$be_babytrain_name
be_chime5_dir=exp/be/$nnet_name/$be_chime5_name
be_ami_dir=exp/be/$nnet_name/$be_ami_name
be_sri_dir=$be_chime5_dir

score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda_slid_win_w${window}_s${period}_gtvad
score_plda_adapt_dir=$score_dir/plda_adapt_slid_win_w${window}_s${period}_gtvad
score_plda_adapt_snorm_dir=$score_dir/plda_adapt_snorm_slid_win_w${window}_s${period}_gtvad

name_vec=(babytrain ami sri)
be_vec=($be_babytrain_dir $be_ami_dir $be_sri_dir)
coh_vec=(jsalt19_spkdet_babytrain_train jsalt19_spkdet_ami_train jsalt19_spkdet_chime5_train)
num_dbs=${#name_vec[@]}
mem_scorer_vec=(30G 10G 10G)

#train_cmd=run.pl

if [ $stage -le 1 ];then

    for((i=0;i<$num_dbs;i++))
    do
	echo "Eval ${name_vec[$i]} with sliding window"
	for part in dev eval
	do
	    db=jsalt19_spkdet_${name_vec[$i]}_${part}
	    coh_data=${coh_vec[$i]}
	    be_dir=${be_vec[$i]}
	    scorer=local/score_${name_vec[$i]}_spkdet.sh
        mem_scorer=${mem_scorer_vec[$i]}

	    for dur in 5 15 30
	    do
        if [ ! -d $xvector_dir/${db}_enr${dur} ];then
            continue
        fi 
		# ground truth diar
		(
		    steps_be/eval_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
					   data/${db}_test/trials/trials_enr$dur \
					   data/${db}_enr${dur}/utt2model \
					   $xvector_dir/${db}_slid_win_w${window}_s${period}_gtvad/utt2orig \
					   $xvector_dir/${db}_enr${dur}_slid_win_w${window}_s${period}_test_track_gtvad/xvector.scp \
					   $be_dir/lda_lnorm.h5 \
					   $be_dir/plda.h5 \
					   $score_plda_dir/${db}_enr${dur}_scores

		    echo '$scorer --cmd "$train_cmd --mem $mem_scorer" \
			    data/${db}_test $part $dur $score_plda_dir'
		    $scorer --cmd "$train_cmd --mem $mem_scorer" \
			    data/${db}_test $part $dur $score_plda_dir
		) &


		# ground truth diar + PLDA adapt
		(
		    steps_be/eval_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
					   data/${db}_test/trials/trials_enr$dur \
					   data/${db}_enr${dur}/utt2model \
					   $xvector_dir/${db}_slid_win_w${window}_s${period}_gtvad/utt2orig \
					   $xvector_dir/${db}_enr${dur}_slid_win_w${window}_s${period}_test_track_gtvad/xvector.scp \
					   $be_dir/lda_lnorm_adapt.h5 \
					   $be_dir/plda_adapt.h5 \
					   $score_plda_adapt_dir/${db}_enr${dur}_scores
		    $scorer --cmd "$train_cmd --mem $mem_scorer" \
			    data/${db}_test $part $dur $score_plda_adapt_dir
		) &

		if [ "$snorm" = true ] ; then
		# ground truth diar + PLDA adapt + AS-Norm
		(
		    steps_be/eval_be_diar_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_type --ncoh $ncoh \
						 data/${db}_test/trials/trials_enr$dur \
						 data/${db}_enr${dur}/utt2model \
                         $xvector_dir/${db}_slid_win_w${window}_s${period}/utt2orig \
						 $xvector_dir/${db}_enr${dur}_slid_win_w${window}_s${period}_test_track/xvector.scp \
						 data/${coh_data}/utt2spk \
						 $xvector_dir/${coh_data}/xvector.scp \
						 $be_dir/lda_lnorm_adapt.h5 \
						 $be_dir/plda_adapt.h5 \
						 $score_plda_adapt_snorm_dir/${db}_enr${dur}_scores
		   $scorer --cmd "$train_cmd --mem 10G" \
		    data/${db}_test $part $dur $score_plda_adapt_snorm_dir
		) &
		fi

	    done
	done
    done
    wait

fi

if [ $stage -le 2 ];then

    for((i=0;i<$num_dbs;i++))
    do
	db=jsalt19_spkdet_${name_vec[$i]}
	scorer=local/score_${name_vec[$i]}_spkdet.sh
    # for plda in plda_gtdiar plda_adapt_gtdiar plda_adapt_snorm_gtdiar
    for plda in plda_slid_win_w${window}_s${period}_gtvad plda_adapt_slid_win_w${window}_s${period}_gtvad
	do

        mem_scorer=${mem_scorer_vec[$i]}
	    for dur in 5 15 30
	    do
        if [ ! -d data/${db}_dev_enr${dur} ];then
           continue
        fi
		(
		    local/calibrate_jsalt19_spkdet_v1.sh --cmd "$train_cmd" \
						 $db $dur $score_dir/$plda

		    $scorer --cmd "$train_cmd --mem $mem_scorer" \
			    data/${db}_dev_test dev $dur $score_dir/${plda}_cal_v1
		    $scorer --cmd "$train_cmd --mem $mem_scorer" \
			    data/${db}_eval_test eval $dur $score_dir/${plda}_cal_v1

		) &
	    done
	done
    done
    wait

fi
