#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
cmd=run.pl
. parse_options.sh || exit 1

if [ $# -ne 4 ]; then
  echo "Usage: $0 <data-root> <dev/eval> <enroll-duration> <score-dir>"
  exit 1;
fi

set -e

data_dir=$1
dev_eval=$2
enr_dur=$3
score_dir=$4

#sri subconditions
if [ "$dev_eval" == "dev" ];then
    conds="clo far tv"
else
    conds="clo med far tv clomed clofarwall"
fi


# keys
trials=$data_dir/trials/trials_enr$enr_dur
trials_sub=$data_dir/trials/trials_sub_enr$enr_dur
db_name=jsalt19_spkdet_sri_enhanced_${dev_eval}

score_file_base=$score_dir/${db_name}_enr${enr_dur}
mkdir -p $score_dir/log
log_file_base=$score_dir/log/${db_name}_enr${enr_dur}

echo "sri enhanced $dev_eval enr-$enr_dur total"
$cmd $log_file_base.log \
     python local/score_dcf.py --key-file $trials \
     --score-file ${score_file_base}_scores \
     --output-path ${score_file_base} &

# echo "sri enhanced $dev_eval enr-$enr_dur subsampled"
# $cmd ${log_file_base}_sub.log \
#      python local/score_dcf.py --key-file $trials_sub \
#      --score-file ${score_file_base}_scores \
#      --output-path ${score_file_base}_sub  &

# for cond_i in $conds
# do
#     echo "sri enhanced $dev_eval enr-$enr_dur $cond_i"
#     $cmd ${log_file_base}_${cond_i}.log \
# 	 python local/score_dcf.py --key-file ${trials}_${cond_i} \
# 	 --score-file ${score_file_base}_scores \
# 	 --output-path ${score_file_base}_${cond_i} &
    
# done

for test_dur in 5 15 30
do
    echo "sri enhanced $dev_eval enr-$enr_dur test-$test_dur"
    $cmd ${log_file_base}_test${test_dur}.log \
	 python local/score_dcf.py --key-file ${trials}_test${test_dur} \
	 --score-file ${score_file_base}_scores \
	 --output-path ${score_file_base}_test${test_dur} &

    # for cond_i in $conds
    # do
    # 	echo "sri $dev_eval enr-$enr_dur test-$test_dur $cond_i"
    # 	$cmd ${log_file_base}_test${test_dur}_${cond_i}.log \
    # 	     python local/score_dcf.py --key-file ${trials}_test${test_dur}_${cond_i} \
    # 	     --score-file ${score_file_base}_scores \
    # 	     --output-path ${score_file_base}_test${test_dur}_${cond_i} &
	
    # done

    
done

wait
