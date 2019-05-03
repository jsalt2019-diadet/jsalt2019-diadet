#! /bin/bash

for name in train "test" dev; do
    rttm=data/$name/rttm
    file=duration_0_$name
    awk '$5 == 0.00' $rttm >> $file 
done

