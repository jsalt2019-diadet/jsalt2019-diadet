#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

cmd=run.pl
prior=0.05

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# -ne 3 ]; then
    echo "Usage: $0 <data-name> <enroll-duration> <score-dir>"
    echo "Ex. $0 jsalt19_spkdet_ami 30 exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda"
  exit 1;
fi

data_name=$1
enr_dur=$2
score_dir=$3

cal_name=${data_name}_enr${enr_dur}
dev_name=${data_name}_dev_enr${enr_dur}
eval_name=${data_name}_eval_enr${enr_dur}
cal_score_dir=${score_dir}_cal_v1

mkdir -p $cal_score_dir

echo "$0 calibrate $cal_name for $score_dir"

model_file=$cal_score_dir/cal_${cal_name}.h5
train_scores=$score_dir/${dev_name}_scores
#we calibrate on test 15 secs condition
train_key=data/${data_name}_dev_test/trials/trials_enr${enr_dur}_test15

$cmd $cal_score_dir/train_cal_${cal_name}.log \
     steps_be/train-calibration-v1.py --score-file $train_scores \
     --key-file $train_key --model-file $model_file --prior $prior

#dev
echo "$0 eval calibration for ${dev_name} for $score_dir"
    
scores_i=${dev_name}_scores
scores_in=$score_dir/$scores_i
scores_out=$cal_score_dir/$scores_i
ndx=data/${data_name}_dev_test/trials/trials_enr${enr_dur}
    $cmd $cal_score_dir/eval_cal_${dev_name}.log \
	 steps_be/eval-calibration-v1.py --in-score-file $scores_in \
	 --ndx-file $ndx --model-file $model_file --out-score-file $scores_out &

#eval
echo "$0 eval calibration for ${eval_name} for $score_dir"
    
scores_i=${eval_name}_scores
scores_in=$score_dir/$scores_i
scores_out=$cal_score_dir/$scores_i
ndx=data/${data_name}_eval_test/trials/trials_enr${enr_dur}
$cmd $cal_score_dir/eval_cal_${eval_name}.log \
     steps_be/eval-calibration-v1.py --in-score-file $scores_in \
     --ndx-file $ndx --model-file $model_file --out-score-file $scores_out &

wait
    # # SITW eval
    # scores_i=sitw_eval_${cond_i}_scores
    # scores_in=$score_dir/$scores_i
    # scores_out=$cal_score_dir/$scores_i
    # ndx=data/sitw_eval_test/trials/${cond_i}.lst
    # $cmd $cal_score_dir/eval_cal_sitw_eval_${cond_i}.log \
    # 	 steps_be/eval-calibration-v1.py --in-score-file $scores_in \
    # 	 --ndx-file $ndx --model-file $model_file --out-score-file $scores_out &


