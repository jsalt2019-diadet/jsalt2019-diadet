#!/bin/bash
# Copyright      2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

score_dir=exp/diarization/$nnet_name/${be_diar_name}
name="$nnet_name $be_diar_name"

score_adapt_dir=exp/diarization/$nnet_name/${be_diar_chime5_name}
score_adapt_reseg_dir=exp/VB/rttm
name_adapt="$nnet_name $be_diar_chime5_name"

#energy VAD
local/make_table_line_spkdiar_jsalt19_chime5.sh --print-header true "$name lstm-vad" $score_dir
local/make_table_line_spkdiar_jsalt19_chime5.sh "$name_adapt lstm-vad" $score_adapt_dir
local/make_table_line_spkdiar_jsalt19_chime5_vb.sh "$name_adapt lstm-vad + reseg" $score_adapt_reseg_dir

echo ""

#GT VAD
local/make_table_line_spkdiar_jsalt19_chime5.sh --print-header true --use-gtvad true "$name" $score_dir
local/make_table_line_spkdiar_jsalt19_chime5.sh --use-gtvad true "$name_adapt" $score_adapt_dir
local/make_table_line_spkdiar_jsalt19_chime5_vb.sh --use-gtvad true "$name_adapt + reseg" $score_adapt_reseg_dir


exit
