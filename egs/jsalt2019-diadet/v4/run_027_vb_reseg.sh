#!/bin/bash
# Copyright 2017-2018  David Snyder
#           2017-2018  Matthew Maciejewski
#           2019 Latané Bullock, Paola Garcia (JSALT 2019) 
#
# Apache 2.0.
#
# This script performs VB resegmentation with a UBM and i-vector 
# extractor as trained in run_026_train_ubm_ive_reseg.sh
# Stages 2 and 3 evaluate the output RTTMs

. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file


be_dir=exp/be_diar/$nnet_name/$be_diar_name
score_dir=exp/diarization/$nnet_name/$be_diar_name

VB_dir=exp/VB
VB_models_dir=$VB_dir/models

#dev datasets
dsets_spkdiar_dev_evad=(jsalt19_spkdiar_babytrain_dev jsalt19_spkdiar_chime5_dev_{U01,U06} jsalt19_spkdiar_ami_dev_{Mix-Headset,Array1-01,Array2-01} jsalt19_spkdiar_sri_dev)
dsets_spkdiar_dev_gtvad=(jsalt19_spkdiar_babytrain_dev_gtvad jsalt19_spkdiar_chime5_dev_{U01,U06}_gtvad jsalt19_spkdiar_ami_dev_{Mix-Headset,Array1-01,Array2-01}_gtvad jsalt19_spkdiar_sri_dev_gtvad) 


#eval datasets
dsets_spkdiar_eval_evad=($(echo ${dsets_spkdiar_dev_evad[@]} | sed 's@_dev@_eval@g'))
dsets_spkdiar_eval_gtvad=($(echo ${dsets_spkdiar_dev_gtvad[@]} | sed 's@_dev@_eval@g'))

dsets_dev=(${dsets_spkdiar_dev_evad[@]} ${dsets_spkdiar_dev_gtvad[@]})
dsets_eval=(${dsets_spkdiar_eval_evad[@]} ${dsets_spkdiar_eval_gtvad[@]})


# consists of all DIAR datasets
dsets_test="${dsets_dev[@]} ${dsets_eval[@]}"


num_components=1024 # the number of UBM components (used for VB resegmentation)
ivector_dim=400 # the dimension of i-vector (used for VB resegmentation)


if [ $stage -le 1 ]; then

  for name in $dsets_test
    do


    # #jobs differ because of the limited number of utterances and 
    # speakers for chime5 - there are just two speakers, so it refuses to split
    # into more than 2
    num_utt=$(wc -l data/$name/utt2spk | cut -d " " -f 1)
	  nj=$(($num_utt < 40 ? 2:40))

    # TODO: turn this into switch?
    if [[ "$name" =~ .*_babytrain_.*_gtvad ]];then
      trained_dir=jsalt19_spkdiar_babytrain_train_gtvad
    elif [[ "$name" =~ .*_ami_.*_gtvad ]];then
      trained_dir=jsalt19_spkdiar_ami_train_gtvad
    elif [[ "$name" =~ .*_chime5_.*_gtvad ]];then
      trained_dir=jsalt19_spkdiar_chime5_train_gtvad
    elif [[ "$name" =~ .*_babytrain_.* ]];then
      trained_dir=jsalt19_spkdiar_babytrain_train
    elif [[ "$name" =~ .*_ami_.* ]];then
      trained_dir=jsalt19_spkdiar_ami_train
    elif [[ "$name" =~ .*_chime5_.* ]];then
      trained_dir=jsalt19_spkdiar_chime5_train
    elif [[ "$name" =~ .*_sri_.* ]];then
      trained_dir=voxceleb2_train_40k
    else
      echo "$name not found"
      exit 1
    fi


    # append _VB to the data dir, then output to the same location
    output_rttm_dir=$score_dir/${name}_VB/plda_scores_tbest
    mkdir -p $output_rttm_dir || exit 1;
    init_rttm_file=$score_dir/$name/plda_scores_tbest/rttm

    # VB resegmentation. In this script, I use the x-vector result to 
    # initialize the VB system. You can also use i-vector result or random 
    # initize the VB system. The following script uses kaldi_io. 
    # You could use `sh ../../../tools/extras/install_kaldi_io.sh` to install it
    # Usage: diarization/VB_resegmentation.sh <data_dir> <init_rttm_filename> <output_dir> <dubm_model> <ie_model>
    VB/diarization/VB_resegmentation.sh --nj $nj --cmd "$train_cmd --mem 10G" \
      --max-iters 1 --initialize 1 \
      data/$name $init_rttm_file $output_rttm_dir \
      $VB_dir/$trained_dir/diag_ubm_$num_components/final.dubm $VB_dir/$trained_dir/extractor_diag_c${num_components}_i${ivector_dim}/final.ie || exit 1; 


    
  done
fi


if [ $stage -le 2 ]; then
  
  for name in $dsets_test
    do

    if [[ "$name" =~ .*_dev.* ]];then
      dev_eval=dev
    elif [[ "$name" =~ .*_eval.* ]];then
      dev_eval=eval
    else
      echo "Dataset dev/eval not found"
      exit 1
    fi

      
    # Compute the DER after VB resegmentation wtih 
    # PYANNOTE
    # "Usage: $0 <dataset> <dev/eval> <score-dir>"
    echo "Starting Pyannote rttm evaluation for $name ... "
    $train_cmd $VB_dir/$name/pyannote.log \
        local/pyannote_score_diar.sh $name $dev_eval $score_dir/${name}_VB/plda_scores_tbest
    

    done

fi





