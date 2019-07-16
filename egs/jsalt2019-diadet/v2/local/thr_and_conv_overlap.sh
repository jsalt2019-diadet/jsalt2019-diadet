#!/bin/bash
# Copyright 2019 JSALT (Diego Castan)  
# Apache 2.0.
#

dataset=${1:-AMI.SpeakerDiarization.MixHeadset}
dest_dir=${2:-./}
onset=${3:-./}
offset=${4:-./}
loadenv=${5:-true}
envname=${6:-'pyannote-audio'}

if $loadenv ; then
source activate ${envname}
fi

# Save file depending on the name
delimiter='.'
s=$dataset$delimiter
array=();
while [[ $s ]]; do
    array+=( "${s%%"$delimiter"*}" );
    s=${s#*"$delimiter"};
done;
name=`echo ${array[0],,}`
outputfile=$dest_dir/overlap_${name}.txt

# Covert the output to txt
echo "Converting"
echo "./local/raw_over2txt.py $dataset $dest_dir --onset $onset --offset $offset --outputfile $outputfile"
./local/raw_over2txt.py $dataset $dest_dir --onset $onset --offset $offset --outputfile $outputfile

