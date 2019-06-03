#!/bin/bash
# Copyright      2019   Johns Hopkins University (Author: Jesus Villalba)
#
# Apache 2.0.
#
# Convert VAD in rttm format to segments format
set -e

frame_shift=0.01

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [  $# != 1 ]; then
    echo "$0 <rttm>"
    exit 1
fi

rttm=$1

awk '$5>="'$frame_shift'" { printf "%s-%07d-%07d %s %.2f %.2f\n",$2,$4,$4+$5,$2,$4,$4+$5}' $rttm 


