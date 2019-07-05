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

data_name=jsalt19_spkdiar_babytrain_enhanced

# Make enhancement data on only eval set
echo "making speech enhancement on $data_name eval"
python local/make_jsalt19_spkdiar.py \
       --list-path $list_path/eval \
       --wav-path $wav_path \
       --output-path $output_path \
       --data-name $data_name \
       --rttm-suffix _test \
       --partition eval

cp $list_path/eval/all_test.uem $output_path/${data_name}_eval/diarization.uem
#make spk2utt so kaldi don't complain
utils/utt2spk_to_spk2utt.pl $output_path/${data_name}_eval/utt2spk \
			    > $output_path/${data_name}_eval/spk2utt
    
utils/fix_data_dir.sh $output_path/${data_name}_eval
