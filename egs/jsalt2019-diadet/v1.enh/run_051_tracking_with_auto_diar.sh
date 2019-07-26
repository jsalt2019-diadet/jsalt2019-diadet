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

score_dir=exp/tracking/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda_${track_diar_name}
score_plda_adapt_dir=$score_dir/plda_adapt_${track_diar_name}
score_plda_adapt_snorm_dir=$score_dir/plda_adapt_snorm_${track_diar_name}
score_plda_gtvad_dir=$score_dir/plda_${track_diar_name}_gtvad
score_plda_adapt_gtvad_dir=$score_dir/plda_adapt_${track_diar_name}_gtvad
score_plda_adapt_snorm_gtvad_dir=$score_dir/plda_adapt_snorm_${track_diar_name}_gtvad

name_vec=(babytrain ami)
be_vec=($be_babytrain_dir $be_ami_dir)
coh_vec=(jsalt19_spkdet_babytrain_train jsalt19_spkdet_ami_train)
num_dbs=${#name_vec[@]}
mem_scorer_vec=(30G 10G)

name_vec=(ami)
be_vec=($be_ami_dir)
coh_vec=(jsalt19_spkdet_ami_train)
num_dbs=${#name_vec[@]}
mem_scorer_vec=(10G)


if [ $stage -le 1 ];then

    for((i=0;i<$num_dbs;i++))
    do
	echo "Eval ${name_vec[$i]} with automatic diarization"
	for part in dev eval
	do
	    db=jsalt19_spkdet_${name_vec[$i]}_${part}
	    coh_data=${coh_vec[$i]}
	    be_dir=${be_vec[$i]}
	    scorer=local/score_${name_vec[$i]}_tracking.sh
	    mem_scorer=${mem_scorer_vec[$i]}

	    for dur in 5 15 30
	    do
		# auto diar + energy VAD
		(
		    steps_be/eval_tracking_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
						 data/${db}_test/trials/trials_enr$dur \
						 data/${db}_enr${dur}/utt2model \
						 data/${db}_test_${track_diar_name}/ext_segments \
						 $xvector_dir/${db}_enr${dur}_test_${track_diar_name}/xvector.scp \
						 $be_dir/lda_lnorm.h5 \
						 $be_dir/plda.h5 \
						 $score_plda_dir/${db}_enr${dur}_scores
		    
		) &

		# auto diar + ground truth VAD
		(
		    steps_be/eval_tracking_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
						 data/${db}_test/trials/trials_enr$dur \
						 data/${db}_enr${dur}/utt2model \
						 data/${db}_test_${spkdet_diar_name}_gtvad/ext_segments \
						 $xvector_dir/${db}_enr${dur}_test_${spkdet_diar_name}_gtvad/xvector.scp \
						 $be_dir/lda_lnorm.h5 \
						 $be_dir/plda.h5 \
						 $score_plda_gtvad_dir/${db}_enr${dur}_scores
		    
		) &

	    done
	done
    done
    wait

fi
