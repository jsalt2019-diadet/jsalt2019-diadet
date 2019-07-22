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
mfccdir=`pwd`/mfcc_enh

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

# Make filterbanks and compute the energy-based VAD for each dataset

if [ $stage -le 1 ]; then
    # Prepare to distribute data over multiple machines
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
	dir_name=$USER/hyp-data/sitw_noisy/v1/$storage_name/mfcc/storage
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

#Train datasets
if [ "$enh_train" == "true" ];then
    if [ $stage -le 2 ];then 
	for name in voxceleb1 voxceleb2_train
	do
	    rm -rf data/${name}_enh${enh_name}
	    cp -r data/$name data/${name}_enh${enh_name}
	    name=${name}_enh${enh_name}
	    steps_pyfe/make_mfcc_enh.sh --write-utt2num-frames true --mfcc-config conf/pymfcc_16k.conf --nj 40 --cmd "$train_cmd" \
	        --chunk-size $enh_chunk_size --nnet-context $enh_context \
		$py_mfcc_enh $enh_nnet data/${name} exp/make_mfcc $mfccdir
	    utils/fix_data_dir.sh data/${name}
	done
    fi


    # Combine voxceleb
    if [ $stage -le 3 ];then 
	utils/combine_data.sh --extra-files "utt2num_frames" data/voxceleb_enh${enh_name} \
	    data/voxceleb1_enh${enh_name} data/voxceleb2_train_enh${enh_name}
	utils/fix_data_dir.sh data/voxceleb_enh${enh_name}

	if [ "$nnet_data" == "voxceleb_div2" ] || [ "$plda_data" == "voxceleb_div2" ];then
	    #divide the size of voxceleb
	    utils/subset_data_dir.sh data/voxceleb_enh${enh_name} $(echo "1236567/2" | bc) data/voxceleb_div2_enh${enh_name}
	fi
    fi
fi

#SITW
if [ $stage -le 4 ];then 
    for name in sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test
    do
	rm -rf data/${name}_enh${enh_name}
	cp -r data/$name data/${name}_enh${enh_name}
	name=${name}_enh${enh_name}
	steps_pyfe/make_mfcc_enh.sh --write-utt2num-frames true --mfcc-config conf/pymfcc_16k.conf --nj 40 --cmd "$train_cmd" \
	                   --chunk-size $enh_chunk_size --nnet-context $enh_context \
			   $py_mfcc_enh $enh_nnet data/${name} exp/make_mfcc $mfccdir
	utils/fix_data_dir.sh data/${name}
    done
fi


