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

xvector_dir=exp/xvectors/$nnet_name
be_babytrain_dir=exp/be/$nnet_name/$be_babytrain_name
be_chime5_dir=exp/be/$nnet_name/$be_chime5_name
be_ami_dir=exp/be/$nnet_name/$be_ami_name
be_sri_dir=$be_chime5_dir

score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda_overlap_${spkdet_diar_name}
score_plda_adapt_dir=$score_dir/plda_adapt_overlap_${spkdet_diar_name}

name_vec=(babytrain ami sri)
be_vec=($be_babytrain_dir $be_ami_dir $be_sri_dir)
num_dbs=${#name_vec[@]}
mem_scorer_vec=(30G 10G 10G)


if [ $stage -le 1 ];then

    for((i=0;i<$num_dbs;i++))
    do
	echo "Eval ${name_vec[$i]} with automatic diarization"
	for part in dev eval
	do
	    db=jsalt19_spkdet_${name_vec[$i]}_${part}
	    be_dir=${be_vec[$i]}
	    scorer=local/score_${name_vec[$i]}_spkdet.sh
	    mem_scorer=${mem_scorer_vec[$i]}

	    for dur in 5 15 30
	    do
		if [ ! -d data/${db}_enr${dur} ];then
		    continue
		fi
		
		# auto diar + energy VAD
		(
		    steps_be/eval_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
						data/${db}_test_overlap/trials/trials_enr$dur \
						data/${db}_enr${dur}/utt2model \
						data/${db}_test_overlap_${spkdet_diar_name}/utt2orig \
						$xvector_dir/${db}_enr${dur}_test_overlap_${spkdet_diar_name}/xvector.scp \
						$be_dir/lda_lnorm.h5 \
						$be_dir/plda.h5 \
						$score_plda_dir/${db}_enr${dur}_scores
		    
		    $scorer --cmd "$train_cmd --mem $mem_scorer" \
			    data/${db}_test_overlap $part $dur $score_plda_dir 
		) &

		# auto diar + energy VAD + PLDA adapt
		(
		    steps_be/eval_be_diar_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
						data/${db}_test_overlap/trials/trials_enr$dur \
						data/${db}_enr${dur}/utt2model \
						data/${db}_test_overlap_${spkdet_diar_name}/utt2orig \
						$xvector_dir/${db}_enr${dur}_test_overlap_${spkdet_diar_name}/xvector.scp \
						$be_dir/lda_lnorm_adapt.h5 \
						$be_dir/plda_adapt.h5 \
						$score_plda_adapt_dir/${db}_enr${dur}_scores
		    
		    $scorer --cmd "$train_cmd --mem $mem_scorer" \
			    data/${db}_test_overlap $part $dur $score_plda_adapt_dir 
		) &

	    done
	done
    done
    wait

fi

##CALIBRATION##

if [ $stage -le 2 ];then

    for((i=0;i<$num_dbs;i++))
    do
	echo "Calibrate scores of ${name_vec[$i]} with automatic diarization"
	db=jsalt19_spkdet_${name_vec[$i]}
    mem_scorer=${mem_scorer_vec[$i]}
	scorer=local/score_${name_vec[$i]}_spkdet.sh
        for plda in plda_overlap_${spkdet_diar_name} plda_adapt_overlap_${spkdet_diar_name} 
	do
	    for dur in 5 15 30
	    do
		if [ ! -d data/${db}_dev_enr${dur} ];then
		    continue
		fi

		(
		    local/calibrate_jsalt19_spkdet_v1.sh --cmd "$train_cmd" \
							 $db $dur $score_dir/$plda
		    $scorer --cmd "$train_cmd --mem $mem_scorer" \
			    data/${db}_dev_test_overlap dev $dur $score_dir/${plda}_cal_v1
		    $scorer --cmd "$train_cmd --mem $mem_scorer" \
			    data/${db}_eval_test_overlap eval $dur $score_dir/${plda}_cal_v1

		) &
	    done
	done
    done
    wait

fi

