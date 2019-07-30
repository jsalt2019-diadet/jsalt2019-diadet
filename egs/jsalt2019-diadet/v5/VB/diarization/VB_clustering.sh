#!/bin/bash
#
# Copyright 2019  Zili Huang
#
#
# This script is a wrapper for Variational Bayes resegmentation.
# It shows how to use the code from Brno University of Technology 
# to do resegmentation.

# Begin configuration section.
nj=20
cmd=run.pl
stage=0
max_speakers=10
max_iters=10
downsample=1
alphaQInit=100.0
sparsityThr=0.001
epsilon=1e-6
minDur=1
loopProb=0
statScale=0.2
llScale=1.0
channel=1
initialize=1
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: diarization/VB_clustering.sh <xvec_dir> <init_labels_filename> <output_dir> <plda_mean> <plda_psi>"
  echo "Variational Bayes Re-segmenatation"
  echo "Options: "
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # How to run jobs."
  echo "  --nj <num-jobs|20>                               # Number of parallel jobs to run."
  echo "  --max-speakers <n|10>                            # Maximum number of speakers" 
  echo "                                                   # expected in the utterance" 
  echo "					           # (default: 10)"
  echo "  --max-iters <n|10>                               # Maximum number of algorithm"
  echo "                                                   # iterations (default: 10)" 
  echo "  --downsample <n|25>                              # Perform diarization on input"
  echo "                                                   # downsampled by this factor"
  echo "                                                   # (default: 25)"
  echo "  --alphaQInit <float|100.0>                       # Dirichlet concentraion"
  echo "                                                   # parameter for initializing q"
  echo "  --sparsityThr <float|0.001>                      # Set occupations smaller that"
  echo "                                                   # this threshold to 0.0 (saves"
  echo "                                                   # memory as the posteriors are"
  echo "                                                   # represented by sparse matrix)"
  echo "  --epsilon <float|1e-6>                           # Stop iterating, if obj. fun." 
  echo "                                                   # improvement is less than" 
  echo "				                   # epsilon"
  echo "  --minDur <n|1>                                   # Minimum number of frames"
  echo "                                                   # between speaker turns imposed"
  echo "                                                   # by linear chains of HMM" 
  echo "                                                   # state corresponding to each" 
  echo "                                                   # speaker. All the states in"
  echo "                                                   # a chain share the same output"
  echo "                                                   # distribution"
  echo "  --loopProb <float|0.9>                           # Probability of not switching"
  echo "                                                   # speakers between frames"
  echo "  --statScale <float|0.2>                          # Scale sufficient statistics" 
  echo "                                                   # collected using UBM"
  echo "  --llScale <float|1.0>                            # Scale UBM likelihood (i.e."
  echo "                                                   # llScale < 1.0 make" 
  echo "                                                   # attribution of frames to UBM"
  echo "                                                   # componets more uncertain)" 
  echo "  --channel <n|0>                                  # Channel information in the rttm file"
  echo "  --initialize <n|1>                               # Whether to initalize the"
  echo "                                                   # speaker posterior (if not)"
  echo "                                                   # the speaker posterior will be"
  echo "                                                   # randomly initilized"

  exit 1;
fi

data_dir=$1
init_labels_filename=$2
output_dir=$3
plda_mean=$4
plda_psi=$5

mkdir -p $output_dir/tmp

#sdata=$data_dir/split$nj;
#utils/split_data.sh $data_dir $nj || exit 1;

if [ $stage -le 0 ]; then
    # VB clustering
    cat $data_dir/utt2spk | awk '{print $2}' | uniq | sort | while read line
    do
	echo $line
	grep $line $data_dir/xvector.txt | cut -d\  -f4-123 > $output_dir/tmpxvec.txt
	if [ `cat $output_dir/tmpxvec.txt | wc -l` -lt 10 ]; then
	    grep $line $init_labels_filename/labels > $output_dir/tmp/$line.label
	    continue
	fi
	
	grep $line $init_labels_filename/labels | awk '{print $2}' > $output_dir/tmplabel.txt
	python3 VB/diarization/VB_clustering.py --max-speakers $max_speakers \
		--max-iters $max_iters --downsample $downsample --alphaQInit $alphaQInit \
		--sparsityThr $sparsityThr --epsilon $epsilon --minDur $minDur \
		--loopProb $loopProb --statScale $statScale --llScale $llScale \
		--channel $channel --initialize $initialize \
		$output_dir/tmpxvec.txt $output_dir/tmplabel.txt $output_dir/tmp/$line.labels $plda_mean $plda_psi || exit 1; 
	sed -i 's/\ /\n/g' $output_dir/tmp/$line.labels
	grep $line $data_dir/utt2spk | paste - $output_dir/tmp/$line.labels | awk '{print $1,$3}' > $output_dir/tmp/$line.label
	rm $output_dir/tmp/$line.labels
    done

    cat $output_dir/tmp/*.label > $output_dir/labels
fi
