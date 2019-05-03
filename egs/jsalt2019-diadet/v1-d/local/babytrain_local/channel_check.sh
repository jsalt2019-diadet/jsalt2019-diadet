#! /bin/bash 

data=$1
log=$2

for audio in $data/*; do
    chnl=$(soxi -c $audio)
    if [ $chnl -gt 1 ]; then
       echo "$audio" >> $log
    fi
done

