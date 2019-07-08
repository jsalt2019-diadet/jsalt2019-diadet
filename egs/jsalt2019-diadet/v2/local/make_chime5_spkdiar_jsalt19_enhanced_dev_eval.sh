#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.

# Make lists for JSALT19 worshop speaker diarization
# for CHiMe5 dataset
set -e 

if [  $# != 3 ]; then
    echo "$0 <wav-path> <list-path> <output_path>"
    exit 1
fi

train_mics="PXX U01 U02 U03 U04 U05 U06"
test_mics="U01 U06"

wav_path=$1
list_path=$2
output_path=$3

data_name=jsalt19_spkdiar_chime5_enhanced


for mic in $test_mics
do
    # Make dev data
    echo "making $data_name dev mic-$mic"
    python local/make_jsalt19_spkdiar.py \
	   --list-path $list_path/dev \
	   --wav-path $wav_path/CHiME5/dev/SE_1000h_model_m3_s3 \
	   --output-path $output_path \
	   --data-name $data_name \
	   --partition dev \
	   --rttm-suffix ${mic}_dev \
	   --mic $mic

    cp $list_path/train/all.$mic.uem $output_path/${data_name}_dev_$mic/diarization.uem
    
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
	   --wav-path $wav_path/CHiME5/test/SE_1000h_model_m3_s3 \
	   --output-path $output_path \
	   --data-name $data_name \
	   --partition eval \
	   --rttm-suffix ${mic}_test \
	   --mic $mic

    
    cp $list_path/eval/all.$mic.uem $output_path/${data_name}_eval_$mic/diarization.uem
    #make spk2utt so kaldi don't complain
    utils/utt2spk_to_spk2utt.pl $output_path/${data_name}_eval_$mic/utt2spk \
				> $output_path/${data_name}_eval_$mic/spk2utt
    
    utils/fix_data_dir.sh $output_path/${data_name}_eval_$mic
done
