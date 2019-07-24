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
    #create subset of voxceleb with segments of more than 15secs
    utils/subset_data_dir.sh \
	--utt-list <(awk '$2>1500' data/$plda_data/utt2num_frames) \
         data/${plda_data} data/${plda_data}_15s

fi


if [ $stage -le 2 ]; then
    # Extract xvectors for training LDA/PLDA
    for name in ${plda_data}_15s
    do
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 60 \
					     $nnet_dir data/${name} \
					     $xvector_dir/${name}
    done

fi

if [ $stage -le 3 ]; then
    # Extracts x-vectors for adaptation data
    for name in jsalt19_spkdet_{babytrain,chime5,ami}{,_enhanced}_train
    do
	num_spk=$(wc -l data/${name}/spk2utt | cut -d " " -f 1)
	nj=$(($num_spk < 40 ? $num_spk:40))
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
					      $nnet_dir data/$name \
					      $xvector_dir/$name
    done
fi


if [ $stage -le 4 ]; then
    # Extracts x-vectors for enrollment
    for db in jsalt19_spkdet_{babytrain,ami,sri}{,_enhanced}_{dev,eval}
    do
	for d in 5 15 30
	do
	    name=${db}_enr$d
	    if [ ! -d data/$name ];then
		continue
	    fi
	    num_spk=$(wc -l data/${name}/spk2utt | cut -d " " -f 1)
	    nj=$(($num_spk < 40 ? $num_spk:40))
	    steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
						 $nnet_dir data/$name \
						 $xvector_dir/$name
	done
    done
fi

if [ $stage -le 5 ]; then
    # Extracts x-vectors for test with ground truth VAD
    for db in jsalt19_spkdet_{babytrain,ami,sri}{,_enhanced}_{dev,eval}
    do
	name=${db}_test_gtvad
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
					     $nnet_dir data/$name \
					     $xvector_dir/$name
    done
fi

if [ $stage -le 6 ]; then
    # Extracts x-vectors for test with energy VAD
    for db in jsalt19_spkdet_{babytrain,ami,sri}{,_enhanced}_{dev,eval}
    do
	name=${db}_test
	steps_kaldi_xvec/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
					     $nnet_dir data/$name \
					     $xvector_dir/$name
    done
fi


if [ $stage -le 7 ]; then
    # combine enroll and test xvectors for step 041
    # no diarization, energy and GT VAD
    for dset in jsalt19_spkdet_{babytrain,ami,sri}{,_enhanced}
    do
	for part in dev eval
	do
	    db=${dset}_${part}
	    for dur in 5 15 30
	    do
		if [ ! -d data/${db}_enr${dur} ];then
		    continue
		fi
		#for energy VAD
		echo "combining ${db}_enr${dur}_test"
		mkdir -p $xvector_dir/${db}_enr${dur}_test
		cat $xvector_dir/${db}_{enr${dur},test}/xvector.scp \
		    > $xvector_dir/${db}_enr${dur}_test/xvector.scp
		
		#for ground truth VAD
		echo "combining ${db}_enr${dur}_test_gtvad"
		mkdir -p $xvector_dir/${db}_enr${dur}_test_gtvad
		cat $xvector_dir/${db}_{enr${dur},test_gtvad}/xvector.scp \
		    > $xvector_dir/${db}_enr${dur}_test_gtvad/xvector.scp
	    done
	done
    done
   
fi


exit
