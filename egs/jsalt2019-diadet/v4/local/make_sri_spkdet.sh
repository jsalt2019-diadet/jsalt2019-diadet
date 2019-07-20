#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.

# Make lists for JSALT19 worshop speaker detection and tracking task
# for SRI dataset
set -e 

if [  $# != 3 ]; then
    echo "$0 <wav-path> <list-path> <output_path>"
    exit 1
fi

wav_path=$1
list_path=$2
output_path=$3

data_name=jsalt19_spkdet_sri

# Make dev data
echo "making $data_name dev"
python local/make_jsalt19_spkdet.py \
       --list-path $list_path/dev \
       --wav-path $wav_path \
       --output-path $output_path \
       --data-name $data_name \
       --partition dev \
       --rttm-suffix _dev \
       --test-dur 60


for d in 30
do
    #make spk2utt so kaldi don't complain
    utils/utt2spk_to_spk2utt.pl $output_path/${data_name}_dev_enr$d/utt2spk \
				> $output_path/${data_name}_dev_enr$d/spk2utt
    
    utils/fix_data_dir.sh $output_path/${data_name}_dev_enr$d
done
cp $output_path/${data_name}_dev_test/utt2spk $output_path/${data_name}_dev_test/spk2utt
utils/fix_data_dir.sh $output_path/${data_name}_dev_test

# cd $output_path/${data_name}_dev_test/trials
# for dur in 30
# do
#     for suffix in "" _test5 _test15 _test30
#     do
# 	awk '$2 ~ /rm4.*mc09/' trials_enr${dur}${suffix} > trials_enr${dur}${suffix}_clo
# 	awk '$2 ~ /rm4.*mc03/ || $2 ~ /rm4.*mc05/ || $2 ~ /rm4.*mc22/' trials_enr${dur}${suffix} > trials_enr${dur}${suffix}_far
# 	awk '$2 ~ /rm4.*mc14/' trials_enr${dur}${suffix} > trials_enr${dur}${suffix}_tv
#     done
# done
# cd -


# Make eval data
echo "making $data_name eval"
python local/make_jsalt19_spkdet.py \
       --list-path $list_path/eval \
       --wav-path $wav_path \
       --output-path $output_path \
       --data-name $data_name \
       --partition eval \
       --rttm-suffix _test \
       --test-dur 60


for d in 30
do
    #make spk2utt so kaldi don't complain
    utils/utt2spk_to_spk2utt.pl $output_path/${data_name}_eval_enr$d/utt2spk \
				> $output_path/${data_name}_eval_enr$d/spk2utt
    
    utils/fix_data_dir.sh $output_path/${data_name}_eval_enr$d
done
cp $output_path/${data_name}_eval_test/utt2spk $output_path/${data_name}_eval_test/spk2utt
utils/fix_data_dir.sh $output_path/${data_name}_eval_test


# #make subconditions trial files
# cd $output_path/${data_name}_eval_test/trials
# for dur in 30
# do
#     for suffix in "" _test5 _test15 _test30
#     do
# 	awk '$2 ~ /rm1.*mc09/' trials_enr${dur}${suffix} \
# 	    > trials_enr${dur}${suffix}_clo
# 	awk '$2 ~ /rm1.*mc05/ || $2 ~ /rm1.*mc16/ || $2 ~ /rm3.*mc18/' \
# 	    trials_enr${dur}${suffix} > trials_enr${dur}${suffix}_med
# 	awk '$2 ~ /rm3.*mc01/' trials_enr${dur}${suffix} \
# 	    > trials_enr${dur}${suffix}_far
# 	awk '$2 ~ /rm1.*mc10/ || $2 ~ /rm2.*mc05/ || $2 ~ /rm2.*mc22/ || $2 ~ /rm3.*mc11/' \
# 	    trials_enr${dur}${suffix} > trials_enr${dur}${suffix}_tv
# 	awk '$2 ~ /rm3.*mc07/ || $2 ~ /rm3.*mc13/ || $2 ~ /rm3.*mc16/' \
# 	    trials_enr${dur}${suffix} > trials_enr${dur}${suffix}_clomed
# 	awk '$2 ~ /rm2.*mc11/ || $2 ~ /rm2.*mc16/ || $2 ~ /rm2.*mc20/' \
# 	    trials_enr${dur}${suffix} > trials_enr${dur}${suffix}_clofarwall
		
#     done
# done
# cd -
