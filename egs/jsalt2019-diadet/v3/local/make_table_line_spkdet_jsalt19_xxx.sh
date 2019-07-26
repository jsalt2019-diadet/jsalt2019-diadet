#!/bin/bash
# Copyright 2019 Johns Hopkins University (Jesus Villalba)  
# Apache 2.0.
#
set -e

print_header=false
enr_durs="30 15 5"
test_durs="30 15 5 0"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
    echo "Usage: $0 <system-name> <babytrain/ami/sri> <score-dir>"
    exit 1;
fi

name=$1
dataset=$2
score_dir=$3

declare -a header_1
declare -a conds
header_2="EER,MinDCF,ActDCF,"

ii=0
for enr_dur in $enr_durs
do
    for test_dur in $test_durs 
    do
	header_1[$ii]="enr=${enr_dur}-test>=${test_dur}"
	if [ "${test_dur}" == "0" ];then
	    conds[ii]=_enr${enr_dur}
	else
	    conds[ii]=_enr${enr_dur}_test${test_dur}
	fi
	ii=$(($ii+1))
    done
done

num_conds=${#conds[*]}
num_cols=$((3*$num_conds))

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
    	printf "${header_1[$i]},,,"
    done
    for((i=0;i<$num_conds;i++));do
    	printf "${header_1[$i]},,,"
    done
    printf "\n"
    
    #print first header
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
	res_file=$score_dir/jsalt19_spkdet_${dataset}_${db}${conds[$i]}_results
	awk '{ printf "%.2f,%.3f,%.3f,", $2,$4,$6}' $res_file
    done
done
printf "\n"
