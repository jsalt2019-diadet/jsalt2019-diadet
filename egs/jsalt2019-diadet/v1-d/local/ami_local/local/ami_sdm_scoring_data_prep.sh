#!/bin/bash

# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski)
#           2016  Johns Hopkins University (Author: Daniel Povey)
#           2018  Matthew Maciejewski
# AMI Corpus dev/eval data preparation
# Apache 2.0

# Note: this is called by ../run.sh.

. ./path.sh

#check existing directories
if [ $# != 3 ]; then
  echo "Usage: $0  <path/to/AMI> <mic-id> <set-name>"
  echo "e.g. $0 /foo/bar/AMI sdm1 dev"
  exit 1;
fi

AMI_DIR=$1
MICNUM=$(echo $2 | sed s/[a-z]//g)
SET=$3
DSET="sdm$MICNUM"

if [ "$DSET" != "$2" ]; then
  echo "$0: bad 2nd argument: $*"
  exit 1
fi

SEGS=data/local/annotations/$SET.txt
tmpdir=data/local/$DSET/$SET
dir=data/$DSET/${SET}

mkdir -p $tmpdir

# Audio data directory check
if [ ! -d $AMI_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1;
fi

# And transcripts check
if [ ! -f $SEGS ]; then
  echo "Error: File $SEGS no found (run ami_text_prep.sh)."
  exit 1;
fi

# find headset wav audio files only, here we again get all
# the files in the corpora and filter only specific sessions
# while building segments

find $AMI_DIR -iname "*.Array1-0$MICNUM.wav" | sort > $tmpdir/wav.flist

n=`cat $tmpdir/wav.flist | wc -l`
echo "In total, $n files were found."

# (1a) Transcriptions preparation
# here we start with normalised transcripts

awk '{meeting=$1; channel="SDM"; speaker=$3; stime=$4; etime=$5;
 printf("AMI_%s_%s_%s_%07.0f_%07.0f", meeting, channel, speaker, int(100*stime+0.5), int(100*etime+0.5));
 for(i=6;i<=NF;i++) printf(" %s", $i); printf "\n"}' $SEGS | sort | uniq > $tmpdir/text

# (1c) Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#AMI_ES2011a_H00_FEE041_0003415_0003484
awk '{
       segment=$1;
       split(segment,S,"[_]");
       audioname=S[1]"_"S[2]"_"S[3]"_"S[4]; startf=S[5]; endf=S[6];
       print segment " " audioname " " startf/100 " " endf/100 " "
}' < $tmpdir/text > $tmpdir/segments

# Make wav.scp
sed -e 's?.*/??' -e 's?.wav??' $tmpdir/wav.flist | \
 perl -ne 'split; $_ =~ m/(.*)\..*/; print "AMI_$1_SDM\n"' | \
  paste - $tmpdir/wav.flist > $tmpdir/wav1.scp

#keep only devset part of waves
awk -F'_' '{print $1"_"$2"_"$3}' $tmpdir/segments | sort -u | join - $tmpdir/wav1.scp > $tmpdir/wav2.scp

#replace path with an appropriate sox command that select single channel only
awk '{print $1" sox -c 1 -t wavpcm -e signed-integer "$2" -t wavpcm - |"}' $tmpdir/wav2.scp > $tmpdir/wav.scp

# Create reco2file_and_channel
cat $tmpdir/wav.scp | \
  perl -ane '$_ =~ m:^(\S+SDM).*\/([IETB].*)\.wav.*$: || die "bad label $_";
       print "$1 $2 A\n"; '\
  > $tmpdir/reco2file_and_channel || exit 1;

# Create reference rttm file for scoring
awk -F'[ _]' '{print "SPEAKER "$1"_"$2"_"$3" 0 "$11" "$12-$11" <NA> <NA> "$7"_"$8"_"$9"_"$10" <NA>"}' \
  $tmpdir/segments > $tmpdir/ref.rttm
#awk -F'[ _]' '{print "SPEAKER "$1"_"$2"_"$3" "$11" "$12-$11" <NA> <NA> "$7"_"$8"_"$9"_"$10" <NA>"}' \
#  $tmpdir/segments > $tmpdir/ref.rttm
#while read line; do
#  sed_command=$(echo "$line" | awk '{print "s/"$1" /"$2" "$3" /g"}')
#  sed -i -e "$sed_command" $tmpdir/ref.rttm
#done < $tmpdir/reco2file_and_channel

# Create reco2num_spk
rm -f $tmpdir/reco2num_spk
for id in $(cut -d' ' -f1 $tmpdir/reco2file_and_channel); do
  num_spk=$(grep -e "$id" $tmpdir/ref.rttm | awk '{print $8}' | sort -u | wc -l)
  echo "$id $num_spk" >> $tmpdir/reco2num_spk
done

# Scrub speaker labels and segmentation from segments
awk -F'[ _]' '{print $1"_"$2"_"$3"_"$5"_"$6" "$1"_"$2"_"$3" "$11" "$12}' $tmpdir/segments > $tmpdir/false_segments
sort $tmpdir/false_segments -o $tmpdir/false_segments
cut -d' ' -f1-2 $tmpdir/false_segments > $tmpdir/false_labels
diarization/make_rttm.py $tmpdir/false_segments $tmpdir/false_labels $tmpdir/false_rttm
awk '{printf("%s_%06d_%06d %s %.2f %.2f\n",$2,$4*100,($4+$5)*100,$2,$4,$4+$5)}' $tmpdir/false_rttm \
  | sort > $tmpdir/segments

# Create utt2spk and spk2utt
awk '{print $1" "$2}' $tmpdir/segments > $tmpdir/utt2spk
sort -k 2 $tmpdir/utt2spk | utils/utt2spk_to_spk2utt.pl > $tmpdir/spk2utt || exit 1;

# Copy stuff into its final locations [this has been moved from the format_data
# script]
mkdir -p $dir
for f in spk2utt utt2spk wav.scp segments reco2file_and_channel reco2num_spk ref.rttm; do
  cp $tmpdir/$f $dir/$f || exit 1;
done

utils/validate_data_dir.sh --no-feats --no-text $dir

echo AMI $DSET scenario and $SET set data preparation succeeded.

