#!/bin/bash
# Copyright
#                2019   Latané Bullock (JSALT 2019)
#
# Apache 2.0.
#


. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file


ovl_out_dir=exp/overlap_models/test_raw
vad_out_dir=exp/vad
vb_out_dir=
ovl_suff=OVLassign

score_dir=exp/diarization/$nnet_name/$be_diar_name

# retrieves all dsets which have had VB resegmentation performed
dsets_path=$score_dir
dsets=`find $dsets_path  -maxdepth 1 -name "jsalt19_spkdiar*VB" | xargs -l basename | sort`

# TESTING(FIXME)
dsets="jsalt19_spkdiar_ami_dev_Mix-Headset_VB"

if [ $stage -le 1 ]; then
	# Given an overlap RTTM, VAD rttm, and the VB-HMM-GMM q matrix (speaker poster attributions), 
	# we perform frame-level reassignment to two speakers in overlap segments

	for dset in $dsets; do
		# We need to do some name parsing
		if [[ "$dset" =~ .*ami.* ]];then
      		corp=ami
    	elif [[ "$dset" =~ .*babytrain.* ]];then
      		corp=babytrain
    	elif [[ "$dset" =~ .*chime5.* ]];then
      		corp=chime5
		elif [[ "$dset" =~ .*sri.* ]];then
      		corp=sri
		else echo "dset -- $dset -- unable to be parsed"; exit 1; 
		fi

		if [[ "$dset" =~ .*dev.* ]];then
      		part=dev
    	elif [[ "$dset" =~ .*eval.* ]];then
      		part=eval
		else echo "dset -- $dset -- unable to be parsed"; exit 1; 
		fi

		# TESTING(FIXME) need to add the rest of the mics
		if [[ "$dset" =~ .*Array1.* ]];then
      		mic=Array1-01
    	elif [[ "$dset" =~ .*Array2.* ]];then
      		mic=Array2-01
    	elif [[ "$dset" =~ .*Mix-Headset.* ]];then
      		mic=Mix-Headset
		else echo "dset -- $dset -- unable to be parsed"; exit 1; 
		fi
		
		
		# This is the dir that the overlap assignment will use 
		output_dir=${dsets_path}/${dset}_${ovl_suff}
		
		mkdir -p $output_dir

		#TESTING(FXIME)
		rm -rf $output_dir/tmp
		rm -rf $output_dir/*.log
		rm -rf $output_dir/result.*
		rm -rf $output_dir/rttm

		# get rttm from BEFORE VB reseg
		# cp $dsets_path/$dset/rttm $output_dir/rttm_in
		
		overlap_rttm=${ovl_out_dir}/diar_overlap_${corp}_${part}_${mic}.rttm
		cat $overlap_rttm > $output_dir/overlap.rttm
		
		# The data dir should contain a pyannoteVAD, utt2spk, and utt2num_frames which are of interest
		data_dir=data/jsalt19_spkdiar_${corp}_${part}_${mic}
		cp $data_dir/utt2spk $output_dir/
		cp $data_dir/utt2num_frames $output_dir/
		
		# We need the q matrix for each utterance from VB resegmentation
		mkdir -p $output_dir/q_mats
		mkdir -p $output_dir/tmp
		# cp $dsets_path/$dset/tmp/*.npy $output_dir/q_mats
		


		# TESTING(FIXME) - this env activation should be placed more strategiclaly 
		source activate pyannote
		
		# Usage: diar_ovl_assignment.py [-h] q_mat_dir ovl_rttm vad_rttm output_dir
		# At this point, the output dir should contain the q matrix from VB reseg
		# an overlap rttm, and a vad rttm 
		python3 local/diar_ovl_assignment.py $output_dir
		cat $output_dir/tmp/*.rttm > $output_dir/rttm
	
	done
fi




# retrieves all which don't already have VB performed 
ovl_dsets=`find $score_dir  -maxdepth 1 -name "jsalt19_spkdiar*" \
  | xargs -l basename \
  | sort \
  | grep "$ovl_suff" \
  `


if [ $stage -le 2 ]; then
  
  for name in $ovl_dsets
  do

    if [[ "$name" =~ .*_dev.* ]];then
      dev_eval=dev
    elif [[ "$name" =~ .*_eval.* ]];then
      dev_eval=eval
    else
      echo "Dataset dev/eval not found"
      exit 1
    fi

    echo $name

    # TESTING(FIXME)
    # we can skip those files which have already been evaluated
    if [ -s $score_dir/$name/result.pyannote-der ];then
      continue
    fi
    

      
    # Compute the DER after VB resegmentation wtih 
    # PYANNOTE
    # "Usage: $0 <dataset> <dev/eval> <score-dir>"
    echo "Starting Pyannote rttm evaluation for $name ... "
    $train_cmd $score_dir/${name}/pyannote.log \
        local/pyannote_score_diar.sh $name $dev_eval $score_dir/${name}
  
    ln -frs $score_dir/${name} $score_dir/${name}/plda_scores_tbest

  done

fi


