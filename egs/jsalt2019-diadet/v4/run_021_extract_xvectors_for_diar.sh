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

dsets_adapt=(jsalt19_spkdiar_{babytrain,chime5,ami}{,_enhanced}_train_gtvad)

dsets_spkdiar_test_evad=(jsalt19_spkdiar_babytrain{,_enhanced}_{dev,eval} jsalt19_spkdiar_chime5{,_enhanced}_{dev,eval}_{U01,U06} jsalt19_spkdiar_ami{,_enhanced}_{dev,eval}_{Mix-Headset,Array1-01,Array2-01})
dsets_spkdiar_test_gtvad=(jsalt19_spkdiar_babytrain{,_enhanced}_{dev,eval}_gtvad jsalt19_spkdiar_chime5{,_enhanced}_{dev,eval}_{U01,U06}_gtvad jsalt19_spkdiar_ami{,_enhanced}_{dev,eval}_{Mix-Headset,Array1-01,Array2-01}_gtvad)

dsets_spkdet_test_evad=(jsalt19_spkdet_babytrain{,_enhanced}_{dev,eval}_test jsalt19_spkdet_ami{,_enhanced}_{dev,eval}_test jsalt19_spkdet_sri{,_enhanced}_{dev,eval}_test jsalt19_spkdet_{ami,sri,babytrain}_{dev,eval}_test_overlap)
dsets_spkdet_test_gtvad=(jsalt19_spkdet_babytrain{,_enhanced}_{dev,eval}_test_gtvad jsalt19_spkdet_ami{,_enhanced}_{dev,eval}_test_gtvad jsalt19_spkdet_sri{,_enhanced}_{dev,eval}_test_gtvad)

#datasets from array to string list
dsets_adapt="${dsets_adapt[@]}"
dsets_test="${dsets_spkdiar_test_evad[@]} ${dsets_spkdiar_test_gtvad[@]} ${dsets_spkdet_test_evad[@]} ${dsets_spkdet_test_gtvad[@]}" 



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

    for name in $dsets_adapt
    do
	utils/subset_data_dir.sh data_diar/${name}_cmn_segmented 50000 data_diar/${name}_cmn_segmented_50k
	num_spk=$(wc -l data_diar/${name}_cmn_segmented_50k/spk2utt | cut -d " " -f 1)
	nj=$(($num_spk < 40 ? $num_spk:40))
	steps_kaldi_diar/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
					     --nj $nj --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false \
					     --hard-min true $nnet_dir \
					     data_diar/${name}_cmn_segmented_50k $xvector_dir/${name}
    done
    
fi


# Extract x-vectors for test datasets
if [ $stage -le 3 ]; then
    for name in $dsets_test
    do
	num_spk=$(wc -l data_diar/${name}_cmn_segmented/spk2utt | cut -d " " -f 1)
	nj=$(($num_spk < 40 ? $num_spk:40))
	steps_kaldi_diar/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
					     --nj $nj --window 1.5 --period 0.75 --apply-cmn false \
					     --min-segment 0.5 $nnet_dir \
					     data_diar/${name}_cmn_segmented $xvector_dir/$name
    done
fi
