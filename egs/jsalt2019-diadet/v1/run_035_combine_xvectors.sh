#!/bin/bash
# Copyright 2019   Johns Hopkins University (Author: Jesus Villalba)
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

xvector_dir=exp/xvectors/$nnet_name

#this script perform some xvector.scp combinations needed for the evaluation

if [ $stage -le 1 ]; then
    # combine enroll and test xvectors for step 041
    # no diarization, energy and GT VAD
    for dset in jsalt19_spkdet_babytrain
    do
	for part in dev eval
	do
	    db=${dset}_${part}
	    for dur in 5 15 30
	    do
		#for energy VAD
		echo "combining ${db}_enr${dur}_test"
		mkdir -p $xvector_dir/${db}_enr${dur}_test
		cat $xvector_dir/${db}_{enr${dur},test}/xvector.scp \
		    > $xvector_dir/${db}_enr${dur}_test/xvector.scp
		
		#for ground truth VAD
		echo "combining ${db}_enr${dur}_test_gtvad"
		mkdir -p $xvector_dir/${db}_enr${dur}_test_gtvad
		cat $xvector_dir/${db}_{enr${dur},test}/xvector.scp \
		    > $xvector_dir/${db}_enr${dur}_test_gtvad/xvector.scp
			 
	    done
	done
    done

fi
exit


if [ $stage -le 2 ]; then

    utils/combine_data.sh data/chime5_spkdet_${diar_name} data/chime5_spkdet_enroll data/chime5_spkdet_test_${diar_name}
    mkdir -p $xvector_dir/chime5_spkdet_${diar_name}
    cat $xvector_dir/chime5_spkdet_{enroll,test_${diar_name}}/xvector.scp > $xvector_dir/chime5_spkdet_${diar_name}/xvector.scp

fi

if [ $stage -le 3 ]; then

    utils/combine_data.sh data/chime5_spkdet_${track_name} data/chime5_spkdet_enroll data/chime5_spkdet_test_${track_name}
    mkdir -p $xvector_dir/chime5_spkdet_${track_name}
    cat $xvector_dir/chime5_spkdet_{enroll,test_${track_name}}/xvector.scp > $xvector_dir/chime5_spkdet_${track_name}/xvector.scp

fi


exit
