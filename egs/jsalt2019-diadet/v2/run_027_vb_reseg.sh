#!/bin/bash
# Copyright 2017-2018  David Snyder
#           2017-2018  Matthew Maciejewski
#           2019 Latané Bullock, Paola Garcia (JSALT 2019) 
#
# Apache 2.0.
#
# This recipe demonstrates the use of x-vectors for speaker diarization.
# The scripts are based on the recipe in ../v1/run.sh, but clusters x-vectors
# instead of i-vectors.  It is similar to the x-vector-based diarization system
# described in "Diarization is Hard: Some Experiences and Lessons Learned for
# the JHU Team in the Inaugural DIHARD Challenge" by Sell et al.  The main
# difference is that we haven't implemented the VB resegmentation yet.

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

#dev datasets
# dsets_spkdiar_dev_evad=(jsalt19_spkdiar_babytrain_dev jsalt19_spkdiar_chime5_dev_{U01,U06} jsalt19_spkdiar_ami_dev_{Mix-Headset,Array1-01,Array2-01} jsalt19_spkdiar_sri_dev)
# dsets_spkdiar_dev_gtvad=(jsalt19_spkdiar_babytrain_dev_gtvad jsalt19_spkdiar_chime5_dev_{U01,U06}_gtvad jsalt19_spkdiar_ami_dev_{Mix-Headset,Array1-01,Array2-01}_gtvad jsalt19_spkdiar_sri_dev_gtvad) 
dsets_spkdiar_dev_evad=(jsalt19_spkdiar_babytrain_dev jsalt19_spkdiar_chime5_dev_{U01,U06} jsalt19_spkdiar_ami_dev_{Mix-Headset,Array1-01,Array2-01} jsalt19_spkdiar_sri_dev)
dsets_spkdiar_dev_gtvad=(jsalt19_spkdiar_babytrain_dev_gtvad jsalt19_spkdiar_ami_dev_{Mix-Headset,Array1-01,Array2-01}_gtvad jsalt19_spkdiar_sri_dev_gtvad) 


#eval datasets
dsets_spkdiar_eval_evad=($(echo ${dsets_spkdiar_dev_evad[@]} | sed 's@_dev@_eval@g'))
dsets_spkdiar_eval_gtvad=($(echo ${dsets_spkdiar_dev_gtvad[@]} | sed 's@_dev@_eval@g'))

dsets_dev=(${dsets_spkdiar_dev_evad[@]} ${dsets_spkdiar_dev_gtvad[@]})
dsets_eval=(${dsets_spkdiar_eval_evad[@]} ${dsets_spkdiar_eval_gtvad[@]})


# consists of all DIAR datasets
dsets_test="${dsets_dev[@]} ${dsets_eval[@]}"

# consists of all DIAR-DEV datasets
# dsets_test="${dsets_dev[@]}"


# consists of all DIAR-GT datasets
# dsets_test="${dsets_dev_gt[@]} ${dsets_eval_gt[@]}"

# consists of all DIAR-EVAL_GT datasets
# dsets_test="${dsets_eval_gt[@]}"

# for testing purposes
dsets_test="jsalt19_spkdiar_sri_dev jsalt19_spkdiar_sri_dev_gtvad jsalt19_spkdiar_sri_eval jsalt19_spkdiar_sri_eval_gtvad"

# subset of all dsets for experimenting with VB max iterations
# vb_iterations_exp_dsets="jsalt19_spkdiar_babytrain_dev jsalt19_spkdiar_chime5_dev_U01 jsalt19_spkdiar_ami_dev_Mix-Headset jsalt19_spkdiar_sri_dev jsalt19_spkdiar_babytrain_dev_gtvad  jsalt19_spkdiar_ami_dev_Mix-Headset_gtvad jsalt19_spkdiar_sri_dev_gtvad"
# vb_iterations_exp_dsets="jsalt19_spkdiar_ami_dev_Mix-Headset_gtvad jsalt19_spkdiar_sri_dev_gtvad"
# dsets_test=$vb_iterations_exp_dsets

echo $dsets_test


num_components=1024 # the number of UBM components (used for VB resegmentation)
# num_components=128 # the number of UBM components (used for VB resegmentation)
ivector_dim=400 # the dimension of i-vector (used for VB resegmentation)
# ivector_dim=50 # the dimension of i-vector (used for VB resegmentation)




if [ $stage -le 1 ]; then

  for name in $dsets_test
    do


    # #jobs differ because of the limited number of utterances and 
    # speakers for chime5 - there are just two speakers, so it refuses to split
    # into more than 2
    if [[ "$name" =~ .*_babytrain_.*_gtvad ]];then
      trained_dir=jsalt19_spkdiar_babytrain_train_gtvad
      nj=20
    elif [[ "$name" =~ .*_ami_.*_gtvad ]];then
      trained_dir=jsalt19_spkdiar_ami_train_gtvad
      nj=20
    elif [[ "$name" =~ .*_chime5_.*_gtvad ]];then
      trained_dir=jsalt19_spkdiar_chime5_train_gtvad
      nj=2
    elif [[ "$name" =~ .*_babytrain_.* ]];then
      trained_dir=jsalt19_spkdiar_babytrain_train
      nj=20
    elif [[ "$name" =~ .*_ami_.* ]];then
      trained_dir=jsalt19_spkdiar_ami_train
      nj=20
    elif [[ "$name" =~ .*_chime5_.* ]];then
      trained_dir=jsalt19_spkdiar_chime5_train
      nj=2
    elif [[ "$name" =~ .*_sri_.* ]];then
      trained_dir=voxceleb2_train_40k
      nj=20    
    else
      echo "$name not found"
      exit 1
    fi


    output_rttm_dir=$VB_dir/$name/vb_iter$vb_niter/rttm
    mkdir -p $output_rttm_dir || exit 1;
    init_rttm_file=$score_dir/$name/plda_scores_tbest/rttm

    # VB resegmentation. In this script, I use the x-vector result to 
    # initialize the VB system. You can also use i-vector result or random 
    # initize the VB system. The following script uses kaldi_io. 
    # You could use `sh ../../../tools/extras/install_kaldi_io.sh` to install it
    # Usage: diarization/VB_resegmentation.sh <data_dir> <init_rttm_filename> <output_dir> <dubm_model> <ie_model>
    VB/diarization/VB_resegmentation.sh --nj $nj --cmd "$train_cmd --mem 10G" \
      --max-iters $vb_niter --initialize 1 \
      data/$name $init_rttm_file $VB_dir/$name/vb_iter$vb_niter \
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
    echo "Starting Pyannote rttm evaluation for $name, vb_iter$vb_niter ... "
    $train_cmd $VB_dir/$name/pyannote.log \
        local/pyannote_score_diar.sh $name $dev_eval $VB_dir/$name/vb_iter$vb_niter/rttm
    

    done

fi



if [ $stage -le 3 ]; then 

  # echo "dset,DER_pre,DER_post,DER_dif,Miss_pre,Miss_post,Miss_diff,FA_pre,FA_post,FA_diff,Conf_pre,Conf_post,Conf_diff,"
  echo "dset,DER_pre,DER_vb_i1,DER_vb_i3,DER_vb_i5,DER_vb_i10"

  for name in $dsets_test
    do 

      pre_res_f=$score_dir/$name/plda_scores_tbest/result.pyannote-der

      # cols=( 2 11 9 13 )  # columns with DER, Miss, FA, Confusion
      cols=( 2 )  # columns with DER, Miss, FA, Confusion
      
      split=(${name//_/,})

      echo -n "$split," 

      for num in ${cols[@]}; do 
        awk -v num=$num '/TOTAL/ { printf "%.2f,", $num}' $pre_res_f
        awk -v num=$num '/TOTAL/ { printf "%.2f,", $num}' $VB_dir/$name/rttm/result.pyannote-der
      done
      echo
     
    done

fi


