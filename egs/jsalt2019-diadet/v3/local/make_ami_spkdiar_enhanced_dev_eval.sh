#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.

# Make lists for JSALT19 worshop speaker diarization
# for AMI dataset
set -e 

if [  $# != 3 ]; then
    echo "$0 <wav-path> <list-path> <output_path>"
    exit 1
fi

train_mics="Mix-Headset Array1-01 Array2-01"
test_mics="Mix-Headset Array1-01 Array2-01"

wav_path=$1
list_path=$2
output_path=$3

data_name=jsalt19_spkdiar_ami_enhanced


for mic in $test_mics
do
    # Make dev data
    echo "making $data_name dev mic-$mic"
    python local/make_jsalt19_spkdiar.py \
	   --list-path $list_path/dev \
	   --wav-path $wav_path/ami/dev/SE_1000h_model_m3_s3 \
	   --output-path $output_path \
	   --data-name $data_name \
	   --partition dev \
	   --mic $mic \
	   --rttm-suffix ${mic}_dev

    awk -v suff="$mic" '{$1=$1"."suff; print $0}' $list_path/train/all.uem \
	> $output_path/${data_name}_dev_$mic/diarization.uem
    
    #make spk2utt so kaldi don't complain
    utils/utt2spk_to_spk2utt.pl $output_path/${data_name}_dev_$mic/utt2spk \
				> $output_path/${data_name}_dev_$mic/spk2utt

    
    utils/fix_data_dir.sh $output_path/${data_name}_dev_$mic

done


for mic in $test_mics
do
    # Make eval data
    echo "making $data_name eval mic-$mic"
    python local/make_jsalt19_spkdiar.py \
	   --list-path $list_path/eval \
	   --wav-path $wav_path/ami/test/SE_1000h_model_m3_s3 \
	   --output-path $output_path \
	   --data-name $data_name \
	   --partition eval \
	   --mic $mic \
	   --rttm-suffix ${mic}_test

    awk -v suff="$mic" '{$1=$1"."suff; print $0}' $list_path/train/all.uem \
	> $output_path/${data_name}_eval_$mic/diarization.uem
    #make spk2utt so kaldi don't complain
    utils/utt2spk_to_spk2utt.pl $output_path/${data_name}_eval_$mic/utt2spk \
				> $output_path/${data_name}_eval_$mic/spk2utt
    
    utils/fix_data_dir.sh $output_path/${data_name}_eval_$mic

done
