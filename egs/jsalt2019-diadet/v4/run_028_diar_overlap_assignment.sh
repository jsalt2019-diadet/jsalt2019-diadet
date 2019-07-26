#!/bin/bash
# Copyright
#            
#
# Apache 2.0.
#
#
#
# JSALT 2019, Latané Bullock



. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file


ovl_out_dir=exp/overlap_models/test_raw
vb_suff=VB
ovl_suff=OVLassign

score_dir=exp/diarization/$nnet_name/$be_diar_name

# Retrieves all dsets which we want to perform resegmentation on
# We first get all dsets which would have a corresponding VB reseg
# Can perform filtering with inverse grep if needed
dsets_path=$score_dir
dsets=`find $dsets_path  -maxdepth 1 -name "jsalt19_spkdiar*" \
  | xargs -l basename \
  | sort \
  | grep -v VB \
  | grep -v sri \
  `



if [ $stage -le 1 ]; then

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
      		mic='_Array1-01'
    	elif [[ "$dset" =~ .*Array2.* ]];then
      		mic='_Array2-01'
    	elif [[ "$dset" =~ .*Mix-Headset.* ]];then
      		mic='_Mix-Headset'
    	elif [[ "$dset" =~ .*U01.* ]];then
      		mic='_U01'
    	elif [[ "$dset" =~ .*U06.* ]];then
      		mic='_U06'
		else mic='';
		fi

		dset_vb=${dset}_${vb_suff}
		if [ ! -d ${dsets_path}/${dset_vb} ]; then 
			continue
		fi
		
		# This is the dir that the overlap assignment will use 
		output_dir=${dsets_path}/${dset_vb}_${ovl_suff}
		
		mkdir -p $output_dir

		#TESTING(FXIME)-clear dir before starting
		rm -rf $output_dir/tmp
		rm -rf $output_dir/*.log
		rm -rf $output_dir/result.*
		rm -rf $output_dir/rttm

		# get rttm from BEFORE VB reseg
		cp $dsets_path/$dset/plda_scores_tbest/rttm $output_dir/rttm_in
		
		overlap_rttm=${ovl_out_dir}/diar_overlap_${corp}_${part}${mic}.rttm
		cat $overlap_rttm > $output_dir/overlap.rttm
		
		# The data dir should contain a utt2spk and utt2num_frames used in VB 
		data_dir=data/jsalt19_spkdiar_${corp}_${part}${mic}
		cp $data_dir/utt2spk $output_dir/
		cp $data_dir/utt2num_frames $output_dir/
		
		# We need the q matrix for each utterance from VB resegmentation
		mkdir -p $output_dir/q_mats
		mkdir -p $output_dir/tmp
		cp $dsets_path/$dset_vb/tmp/*.npy $output_dir/q_mats
		


		# TESTING(FIXME) - this env activation should be placed more strategiclaly 
		source activate pyannote
		
		# Usage: diar_ovl_assignment.py [-h] <overlap_dir>
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
    # we can skip those rttms which have already been evaluated
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


