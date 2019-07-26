#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

print_header=false
use_gtvad=false

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
    echo "Usage: $0 [--print-header <true/false>(default:false) ] [--use-gtvad <true/false>(default:false)] <system-name> <score-dir>"
    exit 1;
fi

name=$1
score_dir=$2

declare -a header_1
declare -a conds
header_2="DER,Miss,FA,Spk-Confusion,"

ii=0
header_dev_1=(All Close Far Close-TV-Radio)
conds_dev=("" ".clo" ".far" ".tv")
header_eval_1=(All Close Medium Far Close-TV-Radio Mix-Close-Med Mix-Close-Far-Wall)
conds_eval=("" ".clo" ".med" ".far" ".tv" ".clomed" ".clofarwall")

if [ "$use_gtvad" == "true" ];then
    vad_suff="_gtvad"
else
    vad_suff=""
fi

dataset="sri"
num_conds_dev=${#conds_dev[*]}
num_cols_dev=$((4*$num_conds_dev))
num_conds_eval=${#conds_eval[*]}
num_cols_eval=$((4*$num_conds_eval))

if [ "$print_header" == "true" ];then
    
    #print database names
    printf "System,${dataset^^} DEV,"
    for((i=0;i<$num_cols_dev-1;i++)); do
	printf ","
    done
    printf "${dataset^^} EVAL,"
    for((i=0;i<$num_cols_eval-1;i++));do
	printf ","
    done
    printf "\n"
    
    #print first header
    printf ","
    for((i=0;i<$num_conds_dev;i++));do
    	printf "${header_dev_1[$i]},,,,"
    done
    for((i=0;i<$num_conds_eval;i++));do
    	printf "${header_eval_1[$i]},,,,"
    done
    printf "\n"
    
    #print second header
    printf ","
    for((i=0;i<$num_conds_dev;i++));do
    	printf "${header_2}"
    done
    for((i=0;i<$num_conds_eval;i++));do
    	printf "${header_2}"
    done
    printf "\n"

fi


printf "$name,"
for((i=0;i<$num_conds_dev;i++))
do
    res_file=$score_dir/jsalt19_spkdiar_sri_dev${vad_suff}/plda_scores_tbest/result${conds_dev[$i]}.pyannote-der
    awk '/TOTAL/ { printf "%.2f,%.2f,%.2f,%.2f,", $2,$11,$9,$13}' $res_file
done
for((i=0;i<$num_conds_eval;i++))
do
    res_file=$score_dir/jsalt19_spkdiar_sri_eval${vad_suff}/plda_scores_tbest/result${conds_eval[$i]}.pyannote-der
    awk '/TOTAL/ { printf "%.2f,%.2f,%.2f,%.2f,", $2,$11,$9,$13}' $res_file
done

printf "\n"
