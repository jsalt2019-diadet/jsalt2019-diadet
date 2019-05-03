#! /bin/bash 
set -eu
src=$1
data=$2
suffix=${3:-""}
duel_chan=${4:-"duel_chan"}
mkdir -p $duel_chan

if [ ! -e $src ]; then
    echo "Usage: $0  path_to_source path_to_data"
    echo "e.g.: $0 baby_train/train data/train"
    exit 1;
fi 

wav=$src/wav
ref=$src/gold
empty_rttm=Empty"$suffix"
 
for utt in $wav/*.wav; do 
    
    if [ ! -f $utt ]; then
        echo "$utt is not a file."
        continue
    fi
    
    merge=false
    uttID=$(basename $utt)
    uttID=${uttID%".wav"}

    if [ ${uttID: -4} = ".tmp" ];then
        continue
    fi 

    rttm=$ref/$uttID."rttm"
    
    cat $rttm > tmp1
    sed 's/\t/ /g' tmp1 >> $data/rttm
    
    # Write reco2num_spk
    cat $rttm > tmp1
    cut -d $'\t' -f 8 < tmp1 > tmp2
    sort -u < tmp2 > tmp3
    num_spk=$(wc -l < tmp3 | cut -d ' ' -f 1)
    
    if [ $num_spk = "0" ]; then 
        echo "$utt" >> $empty_rttm
    else
         echo "$uttID $num_spk" >> $data/reco2num_spk
            
         # Write wav.scp 
         # echo "$uttID $KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 $utt|" >> $data/wav.scp
         # echo "$uttID sox $utt - remix 1,2 |" >> $data/wav.scp 
         # Channel check 
        num_chanl=$(soxi -c $utt) 
        if [ $num_chanl -gt 1 ];then
            sox -v 0.95 $utt $duel_chan/"$uttID.wav" remix 1,2 
            echo "$uttID $duel_chan/$uttID.wav" >> $data/wav.scp
        else
            echo "$uttID $utt" >> $data/wav.scp

        fi
         # Write segments and utt2spk
         # echo "$uttID $uttID" >> $data/utt2spk
         python local/rttm2segment.py $rttm $data
         rm -f rttm2seg.tmp
    fi

    done

rm tmp*
