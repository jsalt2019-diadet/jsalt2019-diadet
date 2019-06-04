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

xvector_dir=exp/xvectors_diar/$nnet_name

tst_datasets="jsalt19_spkdet_babytrain_dev_test jsalt19_spkdet_babytrain_dev_test_gtvad \
		    jsalt19_spkdet_babytrain_eval_test jsalt19_spkdet_babytrain_eval_test_gtvad \
		    jsalt19_spkdiar_babytrain_dev jsalt19_spkdiar_babytrain_dev_gtvad \
		    jsalt19_spkdiar_babytrain_eval jsalt19_spkdiar_babytrain_eval_gtvad"

# Extract x-vectors for train dataset
if [ $stage -le 1 ]; then
  # Reduce the amount of training data for the PLDA,
  utils/subset_data_dir.sh data_diar/${plda_diar_data}_cmn_segmented 128000 data_diar/${plda_diar_data}_cmn_segmented_128k
  # Extract x-vectors for the Voxceleb, which is our PLDA training
  # data.  A long period is used here so that we don't compute too
  # many x-vectors for each recording.
  steps_kaldi_diar/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
				       --nj 40 --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false \
				       --hard-min true $nnet_dir \
				       data_diar/${plda_diar_data}_cmn_segmented_128k $xvector_dir/${plda_diar_data}_128k

fi

# Extract x-vectors for adaptation datasets
if [ $stage -le 2 ]; then

    for name in jsalt19_spkdiar_babytrain_train_gtvad
    do
    
	utils/subset_data_dir.sh data_diar/${name}_cmn_segmented 64000 data_diar/${name}_cmn_segmented_64k
	steps_kaldi_diar/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
					     --nj 40 --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false \
					     --hard-min true $nnet_dir \
					     data_diar/${name}_cmn_segmented_128k $xvector_dir/${name}_128k
    done
    
fi


# Extract x-vectors for test datasets
if [ $stage -le 3 ]; then
    for name in $tst_datasets
    do
	steps_kaldi_diar/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
					     --nj 30 --window 1.5 --period 0.75 --apply-cmn false \
					     --min-segment 0.5 $nnet_dir \
					     data_diar/${name}_cmn_segmented $xvector_dir/$name
    done
fi
