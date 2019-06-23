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




if [ $stage -le 3 ]; then
    # combine enroll and test xvectors for step 043
    # with automatic diarization
    for dset in jsalt19_spkdet_{babytrain,ami}
    do
	for part in dev eval
	do
	    db=${dset}_${part}
	    for dur in 5 15 30
	    do
		echo "combining ${db}_enr${dur}_test"
		mkdir -p $xvector_dir/${db}_enr${dur}_test_${spkdet_diar_name}
		cat $xvector_dir/${db}_{enr${dur},test_${spkdet_diar_name}}/xvector.scp \
		    > $xvector_dir/${db}_enr${dur}_test_${spkdet_diar_name}/xvector.scp
		
.scp
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
