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
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc  # energy VAD
vaddir_gt=`pwd`/vad_gt  # ground truth VAD

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

# Make filterbanks and compute the energy-based VAD for each dataset

if [ $stage -le 1 ]; then
    # Prepare to distribute data over multiple machines
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
	dir_name=$USER/hyp-data/jsalt2019diadet/v4/$storage_name/mfcc/storage
	if [ "$nodes" == "b0" ];then
	    utils/create_split_dir.pl \
			    utils/create_split_dir.pl \
		/export/b{04,05,06,07}/$dir_name $mfccdir/storage
	elif [ "$nodes" == "b1" ];then
	    utils/create_split_dir.pl \
		/export/b{14,15,16,17}/$dir_name $mfccdir/storage
	else 
	    utils/create_split_dir.pl \
		/export/fs01/jsalt19/$dir_name $mfccdir/storage
	fi
    fi
fi



if [ "$enh_train" == false ];then
#Train datasets
if [ $stage -le 2 ];then 
    for name in voxceleb1 voxceleb2_train
    do
	steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_16k.conf --nj 40 --cmd "$train_cmd" \
			   data/${name} exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh data/${name}
	steps_fe/compute_vad_decision.sh --nj 30 --cmd "$train_cmd" \
					 data/${name} exp/make_vad $vaddir
	utils/fix_data_dir.sh data/${name}
    done
fi

# Combine voxceleb
if [ $stage -le 3 ];then 
  utils/combine_data.sh --extra-files "utt2num_frames" data/voxceleb data/voxceleb1 data/voxceleb2_train
  utils/fix_data_dir.sh data/voxceleb

  if [ "$nnet_data" == "voxceleb_div2" ] || [ "$plda_data" == "voxceleb_div2" ];then
      #divide the size of voxceleb
      utils/subset_data_dir.sh data/voxceleb $(echo "1236567/2" | bc) data/voxceleb_div2
  fi
fi
fi

#Spk detection training data
if [ $stage -le 4 ] && [ "$enh_adapt" == false ];then 
    for name in jsalt19_spkdet_{babytrain,chime5,ami}_train 
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
if [ $stage -le 5 ] && [ "$enh_test" == false ];then

    for db in jsalt19_spkdet_{babytrain,ami,sri}_{dev,eval}
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

#Spk diarization train data
if [ $stage -le 6 ] && [ "$enh_adapt" == false ];then 
    for name in jsalt19_spkdiar_{babytrain,chime5,ami}_train
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

#Spk diarization test data
if [ $stage -le 7 ] && [ "$enh_test" == false ];then 
    for name in jsalt19_spkdiar_babytrain_{dev,eval} \
    					  jsalt19_spkdiar_chime5_{dev,eval}_{U01,U06} \
    					  jsalt19_spkdiar_ami_{dev,eval}_{Mix-Headset,Array1-01,Array2-01} 
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


