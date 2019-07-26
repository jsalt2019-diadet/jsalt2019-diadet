#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
. path.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <dataset> <dev/eval> <score-dir>"
  exit 1;
fi

#export PYANNOTE_DATABASE_CONFIG=/export/fs01/jsalt19/databases/database_all.yml
export PYANNOTE_DATABASE_CONFIG=/export/fs01/jsalt19/databases/database.yml

dataset=$1
dev_eval=$2
score_dir=$3

if [ "$dev_eval" == "dev" ];then
    subset=development
else
    subset=test
fi

is_sri=false
if [[ "$dataset" =~ .*babytrain.* ]];then
    task=BabyTrain.SpeakerDiarization.All 
elif [[ "$dataset" =~ .*chime5.*U01.* ]];then
    task=CHiME5.SpeakerDiarization.U01
elif [[ "$dataset" =~ .*chime5.*U06.* ]];then
    task=CHiME5.SpeakerDiarization.U06
elif [[ "$dataset" =~ .*ami.*Mix.* ]];then
    task=AMI.SpeakerDiarization.MixHeadset
elif [[ "$dataset" =~ .*ami.*Array1.* ]];then
    task=AMI.SpeakerDiarization.Array1
elif [[ "$dataset" =~ .*ami.*Array2.* ]];then
    task=AMI.SpeakerDiarization.Array2
elif [[ "$dataset" =~ .*sri.* ]];then
    task=SRI.SpeakerDiarization.All
    is_sri=true
else
    echo "diarization task not found for dataset $dataset in database file"
    exit
fi

conda activate pyannote

echo pyannote-metrics.py diarization --subset=$subset $task $score_dir/rttm 
pyannote-metrics.py diarization --subset=$subset $task $score_dir/rttm \
		    2> $score_dir/pyannote-der.log \
		    > $score_dir/result.pyannote-der &

echo pyannote-metrics.py detection --subset=$subset $task $score_dir/rttm 
pyannote-metrics.py detection --subset=$subset $task $score_dir/rttm \
		    2> $score_dir/pyannote-det.log \
		    > $score_dir/result.pyannote-det &

wait
conda deactivate
exit

if [ "$is_sri" == "true" ];then
    task=SRI.SpeakerDiarization
    if [ "$subset" == "development" ];then
	subtasks="clo far tv"
    else
	subtasks="clo med far tv clomed clofarwall"
    fi
    for subtask in $subtasks
    do
	echo pyannote-metrics.py diarization --subset=$subset $task.$subtask $score_dir/rttm 
	pyannote-metrics.py diarization --subset=$subset $task.$subtask $score_dir/rttm \
			    2> $score_dir/pyannote-der.$subtask.log \
			    > $score_dir/result.$subtask.pyannote-der &
	
	echo pyannote-metrics.py detection --subset=$subset $task.$subtask $score_dir/rttm 
	pyannote-metrics.py detection --subset=$subset $task.$subtask $score_dir/rttm \
			    2> $score_dir/pyannote-det.$subtask.log \
			    > $score_dir/result.$subtask.pyannote-det &
    done
	    

fi

wait

conda deactivate

