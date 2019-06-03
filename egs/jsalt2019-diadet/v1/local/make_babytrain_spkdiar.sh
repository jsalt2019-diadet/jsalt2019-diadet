#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.

# Make lists for JSALT19 worshop speaker diarization
# for Babytrain dataset
set -e 

if [  $# != 3 ]; then
    echo "$0 <wav-path> <list-path> <output_path>"
    exit 1
fi

wav_path=$1
list_path=$2
output_path=$3

data_name=jsalt19_spkdiar_babytrain

# Make Training data
# This will be used to adapt NNet and PLDA models

python local/make_jsalt19_spkdiar.py \
       --list-path $list_path/train \
       --wav-path $wav_path \
       --output-path $output_path \
       --data-name $data_name \
       --partition train

#make spk2utt so kaldi don't complain
utils/utt2spk_to_spk2utt.pl $output_path/${data_name}_train/utt2spk \
			    > $output_path/${data_name}_train/spk2utt

utils/fix_data_dir.sh $output_path/${data_name}_train

# Make dev data
python local/make_jsalt19_spkdiar.py \
       --list-path $list_path/dev \
       --wav-path $wav_path \
       --output-path $output_path \
       --data-name $data_name \
       --partition dev


#make spk2utt so kaldi don't complain
utils/utt2spk_to_spk2utt.pl $output_path/${data_name}_dev/utt2spk \
			    > $output_path/${data_name}_dev/spk2utt
    
utils/fix_data_dir.sh $output_path/${data_name}_dev


# Make eval data
python local/make_jsalt19_spkdiar.py \
       --list-path $list_path/eval \
       --wav-path $wav_path \
       --output-path $output_path \
       --data-name $data_name \
       --partition eval


#make spk2utt so kaldi don't complain
utils/utt2spk_to_spk2utt.pl $output_path/${data_name}_eval/utt2spk \
			    > $output_path/${data_name}_eval/spk2utt
    
utils/fix_data_dir.sh $output_path/${data_name}_eval
