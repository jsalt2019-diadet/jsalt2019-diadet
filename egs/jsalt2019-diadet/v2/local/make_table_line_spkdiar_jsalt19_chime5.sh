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
header_1=(U01 U06)
conds=(_U01 _U06)

if [ "$use_gtvad" == "true" ];then
    vad_suff="_gtvad"
else
    vad_suff=""
fi

dataset="chime5"
num_conds=${#conds[*]}
num_cols=$((4*$num_conds))

if [ "$print_header" == "true" ];then
    
    #print database names
    printf "System,${dataset^^} DEV,"
    for((i=0;i<$num_cols-1;i++)); do
	printf ","
    done
    printf "${dataset^^} EVAL,"
    for((i=0;i<$num_cols-1;i++));do
	printf ","
    done
    printf "\n"
    
    #print first header
    printf ","
    for((i=0;i<$num_conds;i++));do
    	printf "${header_1[$i]},,,,"
    done
    for((i=0;i<$num_conds;i++));do
    	printf "${header_1[$i]},,,,"
    done
    printf "\n"
    
    #print second header
    printf ","
    for((i=0;i<$num_conds;i++));do
    	printf "${header_2}"
    done
    for((i=0;i<$num_conds;i++));do
    	printf "${header_2}"
    done
    printf "\n"

fi


printf "$name,"
for db in dev eval
do
    for((i=0;i<$num_conds;i++))
    do
	res_file=$score_dir/jsalt19_spkdiar_chime5_${db}${conds[$i]}${vad_suff}/plda_scores_tbest/result.pyannote-der
	awk '/TOTAL/ { printf "%.2f,%.2f,%.2f,%.2f,", $2,$11,$9,$13}' $res_file
    done
done
printf "\n"
