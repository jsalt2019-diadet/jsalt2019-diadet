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
nodes=fs01 #by default it puts mfcc in /export/fs01/jsalt19
storage_name=$(date +'%m_%d_%H_%M')
mfccdir=`pwd`/mfcc_speech_enh
vaddir=`pwd`/mfcc_speech_enh  # energy VAD
vaddir_gt=`pwd`/vad_gt  # ground truth VAD

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

# Make MFCC and compute the energy-based VAD for each dataset

#Spk detection training data
if [ $stage -le 1 ];then 
    for name in jsalt19_spkdet_{babytrain,chime5,ami}_enhanced_train 
    do
	steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_16k.conf --nj 40 --cmd "$train_cmd" \
			   data/${name} exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh data/${name}
	#use ground truth VAD
	hyp_utils/rttm_to_bin_vad.sh --nj 5 data/$name/vad.rttm data/$name $vaddir_gt
	utils/fix_data_dir.sh data/${name}
    done
fi

#Spk detection enrollment and test data
if [ $stage -le 2 ];then

    for db in jsalt19_spkdet_{babytrain,ami,sri}_enhanced_{dev,eval}
    do
	# enrollment 
	for d in 5 15 30
	do
	    name=${db}_enr$d
	    if [ ! -d data/$name ];then
		continue
	    fi
	    num_spk=$(wc -l data/$name/spk2utt | cut -d " " -f 1)
	    nj=$(($num_spk < 40 ? $num_spk:40))
	    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_16k.conf --nj $nj --cmd "$train_cmd" \
			       data/${name} exp/make_mfcc $mfccdir
	    utils/fix_data_dir.sh data/${name}
	    if [[ "$db" =~ .*sri.* ]];then
		#for sri we run the energy vad
		steps_fe/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
						 data/${name} exp/make_vad $vaddir
	    else
		# ground truth VAD
		hyp_utils/rttm_to_bin_vad.sh --nj 5 data/$name/vad.rttm data/$name $vaddir_gt
	    fi
	    utils/fix_data_dir.sh data/${name}
	done
	
	# test
	name=${db}_test
	steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_16k.conf --nj 40 --cmd "$train_cmd" \
			   data/${name} exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh data/${name}
	#energy VAD
	steps_fe/compute_vad_decision.sh --nj 30 --cmd "$train_cmd" \
					 data/${name} exp/make_vad $vaddir
	utils/fix_data_dir.sh data/${name}
	
	#create ground truth version of test data
	rm -rf data/${name}_gtvad
	cp -r data/$name data/${name}_gtvad
	hyp_utils/rttm_to_bin_vad.sh --nj 5 data/$name/vad.rttm data/${name}_gtvad $vaddir_gt
	utils/fix_data_dir.sh data/${name}_gtvad
    done
fi

#Spk diarization data
if [ $stage -le 3 ];then 
    for name in jsalt19_spkdiar_babytrain_enhanced_{train,dev,eval} \
    						   jsalt19_spkdiar_chime5_enhanced_train jsalt19_spkdiar_chime5_enhanced_{dev,eval}_{U01,U06} \
    						   jsalt19_spkdiar_ami_enhanced_train jsalt19_spkdiar_ami_enhanced_{dev,eval}_{Mix-Headset,Array1-01,Array2-01} \
						   jsalt19_spkdiar_sri_enhanced_{dev,eval}
    do
	num_utt=$(wc -l data/$name/utt2spk | cut -d " " -f 1)
	nj=$(($num_utt < 40 ? 2:40))
	steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_16k.conf --nj $nj --cmd "$train_cmd" \
			   data/${name} exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh data/${name}
	# energy VAD
	steps_fe/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
					 data/${name} exp/make_vad $vaddir
	utils/fix_data_dir.sh data/${name}

	#create ground truth version of test data
	nj=$(($num_utt < 5 ? 1:5))
	rm -rf data/${name}_gtvad
	cp -r data/$name data/${name}_gtvad
	hyp_utils/rttm_to_bin_vad.sh --nj $nj data/$name/vad.rttm data/${name}_gtvad $vaddir_gt
	utils/fix_data_dir.sh data/${name}_gtvad
    done
fi


