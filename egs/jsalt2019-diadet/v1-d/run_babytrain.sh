#! /bin/bash
#### 2019 Fei Wu, Jiamin Xie, Zili Huang, Paola Garcia,

set -e

stage=0
data_path=data
exp=exp
mfcc=mfcc
vad=mfcc

. ./utils/parse_options.sh
. ./path.sh
. ./cmd.sh

# Data preparation
if [ $stage -le 0 ]; then
    ./local/make_babytrain.sh
fi

# Make MFCC features
if [ $stage -le 1 ]; then
    mkdir -p $mfcc
    mkdir -p $exp

    for name in train "test" dev; do
        steps/make_mfcc.sh --write-utt2num_frames true \
            --mfcc-config conf/mfcc.conf --nj 20 --cmd "$train_cmd"\
            $data_path/$name $exp/make_mfcc $mfcc
        ./utils/fix_data_dir.sh $data_path/$name
    done
fi

if [ $stage -le 2 ]; then
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
        $data_path/train $exp/make_vad $vad
    utils/fix_data_dir.sh $data_path/train

fi

if [ $stage -le 3 ]; then
    for name in train dev "test"; do
        local/prepare_feats.sh --nj 40 --cmd "$train_cmd" \
            $data_path/$name $data_path/${name}_cmn $exp/${name}_cmn
        if [ -f $data_path/$name/vad.scp ]; then
            cp $data_path/$name/vad.scp $data_path/${name}_cm/
        fi
        if [ -f $data_path/$name/segments ]; then
            cp $data_path/$name/segments $data_path/${name}_cmn/
        fi
        utils/fix_data_dir.sh $data_path/${name}_cmn
    done
fi
