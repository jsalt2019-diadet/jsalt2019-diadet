#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Phani Sankar Nidadavolu)
# Apache 2.0.
#
set -e

stage=1
sampling_rate=16000
frame_shift=0.01
rt60s="0.0:0.5 0.5:1.0 1.0:1.5 1.5:4.0"
snrs="15 10 5 0"
snrs_music=""
snrs_noise=""
snrs_chime3bg=""
snrs_babble=""
rirs_info_path=data/rirs_info
mode=train # Train or eval
combine_noises=false
combine_reverbs=false
make_reverb_plus_noise=false
combine_reverb_plus_noises=false
normalize_output=false

echo "$0 $@"  # Print the command line for logging.

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ]; then
    cat >&2 <<EOF
    echo USAGE: $0 [--optional-args] <list of all dirs to augment>
    echo USAGE: $0 --sampling-rate 16000 --snrs \'15 10 5 0\' --mode=train --rirs-info-path data/rirs_info train1 train2
    echo USAGE: $0 --sampling-rate 16000 --snrs \'17 12 7 2\' --rirs-info-path data/rirs_info sitw_eval_test sitw_eval_enroll
    optional-args:
        --sampling-rate <16000>  # Specify the source sampling rate, default:16000
        --snrs  <snr range>      # Specify snr range, defaults: "15 10 5 0"
        --mode <train/val>       # Specify whether the inp dirs are train or eval, default:train
        --rirs-info-path <dir containing the train, eval list partitions> # This will be created in step1
        --stage <int>
        --make-reverb-plus-noise <true/false> # Makes datasets with reverberation plus noise default:false
        --combine-noises <true/false> # combines all noise-type/snr into one datasets default:false
        --combine-reverbs <true/false> # combines all rt60 into one datasets default:false
        --combine-reverb-plus-noises <true/false> # combines all rt60 + noise into one datasets default:false
EOF
    exit 1;
fi

dir_list=$@

if [ -z "$snrs_music" ];then
    snrs_music="$snrs"
fi

if [ -z "$snrs_noise" ];then
    snrs_noise="$snrs"
fi

if [ -z "$snrs_babble" ];then
    snrs_babble="$snrs"
fi

if [ -z "$snrs_chime3bg" ];then
    snrs_chime3bg="$snrs"
fi


echo "Applying augmentation to directories: $dir_list"
echo "SNR range for music is set to $snrs_music"
echo "SNR range for noise is set to $snrs_noise"
echo "SNR range for babble is set to $snrs_babble"
echo "SNR range for ChiMe3+DEMAND is set to $snrs_chime3bg"

# Reverberation opts
sim_rirs_path=RIRS_NOISES/simulated_rirs

# Check if the dirs exist
for d in $dir_list; do
    [ ! -d data/$d ] && echo dir data/$d does not exist && exit 1;
done

if [ $stage -le 1 ]; then
    # Augment the train directories
    for name in $dir_list; do
	if [ -f "data/$name/utt2num_frames" ];then
	    awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/$name/utt2num_frames > data/$name/reco2dur
	else
	    utils/data/get_reco2dur.sh data/$name
	fi
	for snr in $snrs_noise; do
	    # Augment with musan_noise
	    aug_type=noise
	    kaldi_augmentation/augment_data_dir.py --utt-suffix "${aug_type}-snr$snr" --fg-interval 1 \
						   --normalize-output $normalize_output \
						   --fg-snrs "$snr" --fg-noise-dir "data/musan_${aug_type}_${mode}" \
						   --modify-spk-id "false" \
						   data/$name data/${name}_${aug_type}_snr$snr
	    awk -v snr=$snr -v aug=$aug_type '{print $0" "aug" "snr}' data/${name}_${aug_type}_snr$snr/utt2uniq | \
		sort -k1,1 > data/${name}_${aug_type}_snr$snr/utt2snr
	    awk '{print $0" NA NA NA NA"}' data/${name}_${aug_type}_snr$snr/utt2snr > data/${name}_${aug_type}_snr$snr/utt2info
	    utils/fix_data_dir.sh --utt-extra-files "utt2snr utt2info" data/${name}_${aug_type}_snr$snr
	done

	for snr in $snrs_music; do
	    # Augment with musan_music
	    aug_type=music
	    kaldi_augmentation/augment_data_dir.py --utt-suffix "${aug_type}-snr$snr" --bg-snrs "$snr" \
						   --normalize-output $normalize_output \
						   --num-bg-noises "1" --bg-noise-dir "data/musan_${aug_type}_${mode}" \
						   --modify-spk-id "false" \
						   data/$name data/${name}_${aug_type}_snr$snr
	    awk -v snr=$snr -v aug=$aug_type '{print $0" "aug" "snr}' data/${name}_${aug_type}_snr$snr/utt2uniq | \
		sort -k1,1 > data/${name}_${aug_type}_snr$snr/utt2snr
	    awk '{print $0" NA NA NA NA"}' data/${name}_${aug_type}_snr$snr/utt2snr > data/${name}_${aug_type}_snr$snr/utt2info
	    utils/fix_data_dir.sh --utt-extra-files "utt2snr utt2info" data/${name}_${aug_type}_snr$snr
	done

	for snr in $snrs_babble; do
	    # Augment with musan_speech
	    aug_type=babble
	    kaldi_augmentation/augment_data_dir.py --utt-suffix "${aug_type}-snr$snr" --bg-snrs "$snr" \
						   --normalize-output $normalize_output \
						   --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech_${mode}" \
						   --modify-spk-id "false" \
						   data/$name data/${name}_${aug_type}_snr$snr
	    awk -v snr=$snr -v aug=$aug_type '{print $0" "aug" "snr}' data/${name}_${aug_type}_snr$snr/utt2uniq | \
		sort -k1,1 > data/${name}_${aug_type}_snr$snr/utt2snr
	    awk '{print $0" NA NA NA NA"}' data/${name}_${aug_type}_snr$snr/utt2snr > data/${name}_${aug_type}_snr$snr/utt2info
	    utils/fix_data_dir.sh --utt-extra-files "utt2snr utt2info" data/${name}_${aug_type}_snr$snr
	done

	for snr in $snrs_chime3bg; do
	    # Augment with chime3background + demand
	    aug_type=chime3bg
	    kaldi_augmentation/augment_data_dir.py --utt-suffix "${aug_type}-snr$snr" --bg-snrs "$snr" \
						   --normalize-output $normalize_output \
						   --num-bg-noises "1" --bg-noise-dir "data/chime3background_${mode}" \
						   --modify-spk-id "false" \
						   data/$name data/${name}_${aug_type}_snr$snr
	    awk -v snr=$snr -v aug=$aug_type '{print $0" "aug" "snr}' data/${name}_${aug_type}_snr$snr/utt2uniq | \
		sort -k1,1 > data/${name}_${aug_type}_snr$snr/utt2snr
	    awk '{print $0" NA NA NA NA"}' data/${name}_${aug_type}_snr$snr/utt2snr > data/${name}_${aug_type}_snr$snr/utt2info
	    utils/fix_data_dir.sh --utt-extra-files "utt2snr utt2info" data/${name}_${aug_type}_snr$snr
	done
	#   if [ "$mode" == "train" ]; then
	#     # Augment with DEMAND, entire demand is used for training
	#     aug_type=demand
	#     kaldi_augmentation/augment_data_dir.py --utt-suffix "${aug_type}-snr$snr" --bg-snrs "$snr" \
	#         --normalize-output $normalize_output \
	#         --num-bg-noises "1" --bg-noise-dir "data/demand_${mode}" \
	#         --modify-spk-id "false" \
	#         data/$name data/${name}_${aug_type}_snr$snr
	#     awk -v snr=$snr -v aug=$aug_type '{print $0" "aug" "snr}' data/${name}_${aug_type}_snr$snr/utt2uniq | \
	#         sort -k1,1 > data/${name}_${aug_type}_snr$snr/utt2snr
	#     awk '{print $0" NA NA NA NA"}' data/${name}_${aug_type}_snr$snr/utt2snr > data/${name}_${aug_type}_snr$snr/utt2info
	#     utils/fix_data_dir.sh --utt-extra-files "utt2snr utt2info" data/${name}_${aug_type}_snr$snr
	#   fi
	# done
	
	# aug_types="babble noise music chime3bg"
	# if [ "$mode" == "train" ]; then
	#     aug_types=$aug_types" demand"
	# fi
	
	if [ "$combine_noises" == "true" ];then
	    # Also combine different snr directories to a single directory for each aug type
	    for aug_type in $aug_types; do
		combine_str=""
		for snr in $snrs; do
		    combine_str=$combine_str" data/${name}_${aug_type}_snr$snr"
		done
		utils/combine_data.sh --extra-files "utt2snr utt2info" \
				      data/${name}_${aug_type} $combine_str
	    done
	fi
    done
    
fi

if [ $stage -le 2 ]; then
    # Reverberate speech using RIRs in the range min < rt60 < max for train dirs
    for name in $dir_list; do
	combine_str=""
	for rt60_range in $rt60s; do
	    # Reverberate speech using RIRs in the range 0.0 < rt60 < 0.5
	    rt60_min=`echo $rt60_range | cut -d ":" -f1`
	    rt60_max=`echo $rt60_range | cut -d ":" -f2`
	    kwrd=rt60_min_${rt60_min}_max_${rt60_max}
	    kwrds=" $kwrd"
	    name_aug=${name}_reverb_${kwrd}
	    suff_aug="-reverb-rt60-${rt60_min}-${rt60_max}"
	    combine_str=$combine_str" data/${name_aug}"

	    # Make a version with reverberated speech
	    rvb_opts=()
	    rvb_opts+=(--rir-set-parameters "1.0, $rirs_info_path/rir_list_${mode}_${kwrd}")
	    
	    rm -rf data/${name_aug}
	    
	    # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
	    # additive noise here.
	    kaldi_augmentation/reverberate_data_dir.py \
		"${rvb_opts[@]}" \
		--speech-rvb-probability 1 \
		--pointsource-noise-addition-probability 0 \
		--isotropic-noise-addition-probability 0 \
		--num-replications 1 \
		--normalize-output $normalize_output \
		--source-sampling-rate $sampling_rate \
		data/$name data/${name_aug}
	    cp data/$name/vad.scp data/${name_aug}
	    utils/copy_data_dir.sh --utt-suffix ${suff_aug} data/${name_aug} data/${name_aug}.new
	    rm -rf data/${name_aug}
	    mv data/${name_aug}.new data/${name_aug}

	    awk -v suff=$suff_aug '{ $1=$1""suff; print $0}' data/${name}/reco2dur > data/${name_aug}/reco2dur
	    if [ -f "data/$name/utt2num_frames" ];then
		awk -v suff=$suff_aug '{ $1=$1""suff; print $0}' data/${name}/utt2num_frames > data/${name_aug}/utt2num_frames
	    fi
	    
	    # Create utt2rt60 file
	    python kaldi_augmentation/make_utt2reverb_info_ver2.py data/${name_aug} $rirs_info_path/simrirs2rt60.map \
		   data/${name_aug}/utt2reverbinfo || exit 1;
	    # To create utt2info file for reverb only directories we are using an aribitrary snr value of 40db and aug type is set to None
	    awk '{print $1" "$2" None 40 "$3" "$4" "$5" "$6}' data/${name_aug}/utt2reverbinfo > data/${name_aug}/utt2info
	done
	if [ "$combine_reverbs" == "true" ];then
	    utils/combine_data.sh --extra-files "utt2reverbinfo utt2info" data/${name}_reverb $combine_str
	fi
    done
fi


if [ $stage -le 3 ] && [ "$make_reverb_plus_noise" == "true" ]; then
    # Add noise to reverberated speech from step 2
    for name0 in $dir_list; do
	combine_str=""
	for rt60_range in $rt60s; do
	    rt60_min=`echo $rt60_range | cut -d ":" -f1`
	    rt60_max=`echo $rt60_range | cut -d ":" -f2`
	    kwrd_reverb=rt60_min_${rt60_min}_max_${rt60_max}
	    name=${name0}_reverb_${kwrd_reverb}

	    for snr in $snrs_noise; do
		# Augment with musan_noise
		aug_type=noise
		name_aug=${name}_${aug_type}_snr${snr}
		suff_aug=${aug_type}-snr$snr
		kaldi_augmentation/augment_data_dir.py --utt-suffix "${suff_aug}" --fg-interval 1 \
						       --normalize-output $normalize_output \
						       --fg-snrs "$snr" --fg-noise-dir "data/musan_${aug_type}_${mode}" \
						       --modify-spk-id "false" \
						       data/$name data/${name_aug}
		awk -v snr=$snr -v aug=$aug_type '{print $0" "aug" "snr}' data/${name_aug}/utt2uniq | \
		    sort -k1,1 > data/${name_aug}/utt2snr
		awk -v suff="${suff_aug}" -v snr=$snr -v aug=$aug_type \
		    '{ $1=$1"-"suff; $3=aug; $4=snr; print $0}' data/${name}/utt2info > data/${name_aug}/utt2info
		utils/fix_data_dir.sh --utt-extra-files "utt2snr utt2info" data/${name_aug}
		combine_str="$combine_str data/${name_aug}"
	    done
	    
	    for snr in $snrs_music; do
		# Augment with musan_music
		aug_type=music
		name_aug=${name}_${aug_type}_snr${snr}
		suff_aug=${aug_type}-snr$snr
		kaldi_augmentation/augment_data_dir.py --utt-suffix "${suff_aug}" --bg-snrs "$snr" \
						       --normalize-output $normalize_output \
						       --num-bg-noises "1" --bg-noise-dir "data/musan_${aug_type}_${mode}" \
						       --modify-spk-id "false" \
						       data/$name data/${name_aug}
		awk -v snr=$snr -v aug=$aug_type '{print $0" "aug" "snr}' data/${name_aug}/utt2uniq | \
		    sort -k1,1 > data/${name_aug}/utt2snr
		awk -v suff="${suff_aug}" -v snr=$snr -v aug=$aug_type \
		    '{ $1=$1"-"suff; $3=aug; $4=snr; print $0}' data/${name}/utt2info > data/${name_aug}/utt2info
		
		utils/fix_data_dir.sh --utt-extra-files "utt2snr utt2info" data/${name_aug}
		combine_str="$combine_str data/${name_aug}"
	    done
	    
	    for snr in $snrs_babble; do
		# Augment with musan_speech
		aug_type=babble
		name_aug=${name}_${aug_type}_snr${snr}
		suff_aug=${aug_type}-snr$snr
		kaldi_augmentation/augment_data_dir.py --utt-suffix "${suff_aug}" --bg-snrs "$snr" \
						       --normalize-output $normalize_output \
						       --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech_${mode}" \
						       --modify-spk-id "false" \
						       data/$name data/${name_aug}
		awk -v snr=$snr -v aug=$aug_type '{print $0" "aug" "snr}' data/${name_aug}/utt2uniq | \
		    sort -k1,1 > data/${name_aug}/utt2snr
		awk -v suff="${suff_aug}" -v snr=$snr -v aug=$aug_type \
		    '{ $1=$1"-"suff; $3=aug; $4=snr; print $0}' data/${name}/utt2info > data/${name_aug}/utt2info
		combine_str="$combine_str data/${name_aug}"
		utils/fix_data_dir.sh --utt-extra-files "utt2snr utt2info" data/${name_aug}
	    done
	    
	    for snr in $snrs_chime3bg; do
		# Augment with chime3background
		aug_type=chime3bg
		name_aug=${name}_${aug_type}_snr${snr}
		suff_aug=${aug_type}-snr$snr
		kaldi_augmentation/augment_data_dir.py --utt-suffix "${suff_aug}" --bg-snrs "$snr" \
						       --normalize-output $normalize_output \
						       --num-bg-noises "1" --bg-noise-dir "data/chime3background_${mode}" \
						       --modify-spk-id "false" \
						       data/$name data/${name_aug}
		awk -v snr=$snr -v aug=$aug_type '{print $0" "aug" "snr}' data/${name_aug}/utt2uniq | \
		    sort -k1,1 > data/${name_aug}/utt2snr
		awk -v suff="${suff_aug}" -v snr=$snr -v aug=$aug_type \
		    '{ $1=$1"-"suff; $3=aug; $4=snr; print $0}' data/${name}/utt2info > data/${name_aug}/utt2info
		
		utils/fix_data_dir.sh --utt-extra-files "utt2snr utt2info" data/${name_aug}
		combine_str="$combine_str data/${name_aug}"
		
	    done

	done
	if [ "$combine_reverb_plus_noises" == "true" ];then
	    utils/combine_data.sh --extra-files "utt2snr utt2reverbinfo utt2info" data/${name}_reverb_noise $combine_str
	fi
    done
    
fi
