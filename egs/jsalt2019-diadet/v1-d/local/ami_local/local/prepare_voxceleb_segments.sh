#!/bin/bash

srcdir=$1
datadir=$2

rm -f $datadir/segments

for spk in $(ls -F $srcdir/voxceleb1 | grep /); do
  for file in $(ls $srcdir/voxceleb1/$spk); do
    idspk=$(echo "$spk" | sed -e s/[^A-Za-z_]//g)
    id="${file%.*}"
    awk -v spk="$idspk" -v id="$id" '{if (NR > 5) {printf("%s-%s_%06d_%06d %s-%s %.2f %.2f\n",spk,id,$1*100,$2*100,spk,id,$1,$2)}}' \
        $srcdir/voxceleb1/$spk/$file >> $datadir/segments
  done
done

awk -F'[- ]' '{print $1"-"$2" "$1}' $datadir/segments > $datadir/utt2spk
utils/utt2spk_to_spk2utt.pl $datadir/utt2spk > $datadir/spk2utt
