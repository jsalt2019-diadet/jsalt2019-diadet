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
score_plda_dir=$score_dir/plda_trackgtdiar
score_plda_adapt_dir=$score_dir/plda_adapt_trackgtdiar
score_plda_adapt_snorm_dir=$score_dir/plda_adapt_snorm_trackgtdiar

name_vec=(babytrain ami)
be_vec=($be_babytrain_dir $be_ami_dir)
coh_vec=(jsalt19_spkdet_babytrain_train jsalt19_spkdet_ami_train)
num_dbs=${#name_vec[@]}

#train_cmd=run.pl

if [ $stage -le 1 ];then

    for((i=0;i<$num_dbs;i++))
    do
	echo "Eval ${name_vec[$i]} with ground truth diarization"
	for part in dev eval
	do
	    db=jsalt19_spkdet_${name_vec[$i]}_${part}
	    coh_data=${coh_vec[$i]}
	    be_dir=${be_vec[$i]}
	    scorer=local/score_${name_vec[$i]}_tracking.sh

	    for dur in 5 15 30
	    do
		# ground truth diar
		(
		    steps_be/eval_tracking_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
					   data/${db}_test/trials/trials_enr$dur \
					   data/${db}_enr${dur}/utt2model \
					   data/${db}_test_trackgtdiar/ext_segments \
					   $xvector_dir/${db}_enr${dur}_test_trackgtdiar/xvector.scp \
					   $be_dir/lda_lnorm.h5 \
					   $be_dir/plda.h5 \
					   $score_plda_dir/${db}_enr${dur}_rttm
		    
		    $scorer data/${db}_test/trials $part $dur $score_plda_dir 
		) #&


		# ground truth diar + PLDA adapt
		(
		    steps_be/eval_tracking_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
					   data/${db}_test/trials/trials_enr$dur \
					   data/${db}_enr${dur}/utt2model \
					   data/${db}_test_trackgtdiar/ext_segments \
					   $xvector_dir/${db}_enr${dur}_test_trackgtdiar/xvector.scp \
					   $be_dir/lda_lnorm_adapt.h5 \
					   $be_dir/plda_adapt.h5 \
					   $score_plda_adapt_dir/${db}_enr${dur}_rttm
		    
		    $scorer data/${db}_test/trials $part $dur $score_plda_adapt_dir 
		) #&

		# # ground truth diar + PLDA adapt + AS-Norm
		# (
		#     steps_be/eval_tracking_snorm_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
		# 				 data/${db}_test/trials/trials_enr$dur \
		# 				 data/${db}_enr${dur}/utt2model \
		# 				 data/${db}_test_trackgtdiar/ext_segments \
		# 				 $xvector_dir/${db}_enr${dur}_test_trackgtdiar/xvector.scp \
		# 				 data/${coh_data}/utt2spk \
		# 				 $xvector_dir/${coh_data}/xvector.scp \
		# 				 $be_dir/lda_lnorm_adapt.h5 \
		# 				 $be_dir/plda_adapt.h5 \
		# 				 $score_plda_adapt_snorm_dir/${db}_enr${dur}_rttm
		    
		#     $scorer data/${db}_test/trials $part $dur $score_plda_adapt_snorm_dir 
		# ) #&


	    done
	done
    done
    wait

fi
exit
if [ $stage -le 2 ];then

    for((i=0;i<$num_dbs;i++))
    do
	db=jsalt19_spkdet_${name_vec[$i]}

        for plda in plda_trackgtdiar plda_adapt_trackgtdiar plda_adapt_snorm_trackgtdiar
	do
	    for dur in 5 15 30
	    do
		(
		    local/calibrate_${name_vec[$i]}_tracking_v1.sh --cmd "$train_cmd" $dur $score_dir/$plda
		    $scorer data/${db}_test/trials dev $dur $score_dir/${plda}_cal_v1
		    $scorer data/${db}_test/trials eval $dur $score_dir/${plda}_cal_v1
		) &
	    done
	done
    done
    wait

fi




#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

net_name=3b
diar_name=track3b_t-0.9

lda_dim=300
plda_y_dim=175
plda_z_dim=200

stage=1

. parse_options.sh || exit 1;


xvector_dir=exp/xvectors/$net_name

plda_data=train_combined
plda_type=splda
plda_label=${plda_type}y${plda_y_dim}_v1

be_name=lda${lda_dim}_${plda_label}_${plda_data}
be_dir=exp/be/$net_name/$be_name

score_dir=exp/scores/$net_name/${be_name}
score_plda_dir=$score_dir/plda_${diar_name}


if [ $stage -le 1 ]; then

    echo "Chime5 tracking ${diar_name}"
    steps_be/eval_tracking_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
			   data/chime5_spkdet_test/trials_tracking \
			   data/chime5_spkdet_enroll/utt2spk \
			   data/chime5_spkdet_test_${diar_name}/ext_segments \
			   $xvector_dir/chime5_spkdet_${diar_name}/xvector.scp \
			   $be_dir/lda_lnorm.h5 \
			   $be_dir/plda.h5 \
			   $score_plda_dir/chime5_spkdet_rttm
    
    #local/score_chime5_tracking.sh data/chime5_spkdet_test $score_plda_dir &

fi
exit

if [ $stage -le 2 ];then
    local/calibrate_chime5_spkdet_v1.sh --cmd "$train_cmd" $score_plda_dir 
    local/score_chime5_spkdet.sh data/chime5_spkdet_test ${score_plda_dir}_cal_v1 
    
fi

    
exit
