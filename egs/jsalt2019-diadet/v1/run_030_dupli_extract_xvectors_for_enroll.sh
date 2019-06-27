#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
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
if [ $stage -le 1 ]; then
    # Extracts x-vectors for enrollment
    for db in jsalt19_spkdet_{babytrain,ami}_{dev,eval}
    do
	for d in 5 15 30
	do
	    name=${db}_enr$d
	    num_spk=$(wc -l data/${name}/spk2utt | cut -d " " -f 1)
	    nj=$(($num_spk < 40 ? $num_spk:40))
	    steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
						 $nnet_dir data/$name \
						 $xvector_dir/$name
	done
    done
fi

exit
