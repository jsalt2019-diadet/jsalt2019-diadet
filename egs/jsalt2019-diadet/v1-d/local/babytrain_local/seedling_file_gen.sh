#! /bin/bash 

src=$1
data=$2
suffix=${3:-""}

if [ ! -e $src ]; then
    echo "Usage: $0  path_to_source path_to_data."
    echo "e.g.: $0 /export/corpora/LDC/LDC2018E32v1.1 data_seedlings"
    exit 1;
fi 

flac=$src/data/flac
sad=$src/data/sad
rttm=$src/data/rttm
sources=$src/docs/sources.tbl

for utt in $flac/*.flac; do 
    uttID=$(basename $utt)
    uttID=${uttID%".flac"}
    
    uttID_disamb=$uttID"_$suffix"

    grep $uttID $sources > tmp
    corpus=$(cut -f 3 < tmp)

    if [ $corpus == "SEEDLINGS" ]; then
        # Write rttm
        cat $rttm/$uttID".rttm" > tmp0
        sed "s/$uttID/$uttID_disamb/g" tmp0 > tmp1
        cat tmp1 >> $data/rttm
        cat $rttm/$uttID".rttm" > tmp1
        
        # Write reco2num_spk 
        cut -d ' ' -f 8 < tmp1 > tmp2
        sort -u < tmp2 > tmp3
        num_spk=$(wc -l < tmp3 | cut -d ' ' -f 1)
        echo "$uttID_disamb $num_spk" >> $data/reco2num_spk
       
        # Write wav.scp
        echo  "$uttID_disamb sox -t flac $utt -t wav -r 16k -b 16 - channels 1|" >> $data/wav.scp
        
        # Write segments and utt2spk
        seg_cnt=0
        while read -r line; do 
            seg_start=$(echo $line | cut -d ' ' -f1 )
            seg_end=$(echo $line | cut -d ' ' -f2 )
            seg_cnt=$((seg_cnt + 1))
            printf -v tmp "%04d" $seg_cnt
            seg_ID=$uttID_disamb"_$tmp"
            echo "$seg_ID $uttID_disamb $seg_start $seg_end" >> $data/segments
            echo "$seg_ID $uttID_disamb" >> $data/utt2spk
        done < $sad/$uttID.lab

    fi
done

rm tmp*
