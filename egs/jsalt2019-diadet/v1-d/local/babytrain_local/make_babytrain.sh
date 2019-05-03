#! /bin/bash

dev=baby_train/dev
test=baby_train/test
train=baby_train/train

data=data
mkdir -p $data

duel_chan=duel_chan
mkdir -p $duel_chan

for dir in dev "test" train; do
# for dir in dev; do
    mkdir -p $data/$dir
    ./local/FileCheck.sh $data/$dir
    
    touch $data/$dir/segments
    touch $data/$dir/utt2spk

    ./local/babytrain_file_gen.sh baby_train/$dir $data/$dir "_$dir" $duel_chan
    utils/utt2spk_to_spk2utt.pl $data/$dir/utt2spk > $data/$dir/spk2utt
    utils/fix_data_dir.sh $data/$dir
done


