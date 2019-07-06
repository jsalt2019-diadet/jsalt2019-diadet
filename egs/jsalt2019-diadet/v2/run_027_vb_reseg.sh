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
dsets_spkdiar_dev_evad=( jsalt19_spkdiar_sri_dev )
dsets_spkdiar_dev_gtvad=( jsalt19_spkdiar_sri_dev_gtvad ) 


#eval datasets
dsets_spkdiar_eval_evad=($(echo ${dsets_spkdiar_dev_evad[@]} | sed 's@_dev@_eval@g'))
dsets_spkdiar_eval_gtvad=($(echo ${dsets_spkdiar_dev_gtvad[@]} | sed 's@_dev@_eval@g'))

dsets_dev=(${dsets_spkdiar_dev_evad[@]} ${dsets_spkdiar_dev_gtvad[@]})
dsets_eval=(${dsets_spkdiar_eval_evad[@]} ${dsets_spkdiar_eval_gtvad[@]})


# consists of all DIAR datasets
dsets_test="${dsets_dev[@]} ${dsets_eval[@]}"

# consists of all DIAR-GT datasets
# dsets_test="${dsets_dev_gt[@]} ${dsets_eval_gt[@]}"

# consists of all DIAR-EVAL_GT datasets
# dsets_test="${dsets_eval_gt[@]}"

# for testing purposes
# dsets_test=jsalt19_spkdiar_ami_dev_Array1-01

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

    output_rttm_dir=$VB_dir/$name/rttm
    mkdir -p $output_rttm_dir || exit 1;
    init_rttm_file=$score_dir/$name/plda_scores_tbest/rttm

    # VB resegmentation. In this script, I use the x-vector result to 
    # initialize the VB system. You can also use i-vector result or random 
    # initize the VB system. The following script uses kaldi_io. 
    # You could use `sh ../../../tools/extras/install_kaldi_io.sh` to install it
    VB/diarization/VB_resegmentation.sh --nj $nj --cmd "$train_cmd --mem 10G" \
      --initialize 1 data/$name $init_rttm_file $VB_dir/$name \
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
    echo "Starting Pyannote rttm evaluation for $name ... "
    $train_cmd $VB_dir/$name/pyannote.log \
        local/pyannote_score_diar.sh $name $dev_eval $VB_dir/$name/rttm
  
    
    # # Compute the DER after VB resegmentation wtih 
    # # MD-EVAL
    # mkdir -p $VB_dir/$name/rttm || exit 1;
    # md-eval.pl -1 -r data/$name/diarization.rttm\
    #   -s $VB_dir/$name/rttm/VB_rttm 2> $VB_dir/$name/log/VB_DER.log \
    #   > $VB_dir/$name/rttm/results.md-eval
    # der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    #   $VB_dir/$name/rttm/results.md-eval)
    # pre_der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    #   $VB_dir/$name/rttm/pre_result.md-eval)
    # echo "$name :   DER (pre_VB, post_VB):   $pre_der, $der%"
    
    done

fi

if [ $stage -le 3 ]; then 

  echo "dset,DER_pre,DER_post,DER_dif,Miss_pre,Miss_post,Miss_diff,FA_pre,FA_post,FA_diff,Conf_pre,Conf_post,Conf_diff,"

  for name in $dsets_test
    do 

      post_res_f=$VB_dir/$name/rttm/result.pyannote-der
      pre_res_f=$score_dir/$name/plda_scores_tbest/result.pyannote-der

      cols=( 2, 11, 9, 13 )  # columns with DER, Miss, FA, Confusion
      line="$name,"
      echo -n $line 
      for num in ${cols[@]}; do 
        awk -v num=$num '/TOTAL/ { printf "%.2f,", $num}' $pre_res_f
        awk -v num=$num '/TOTAL/ { printf "%.2f,", $num}' $post_res_f
        # line="$line (awk -v num=$num'/TOTAL/ { printf \"%.2f,,\", $num}' $pre_res_f)"
        # line="$line (awk '/TOTAL/ { printf \"%.2f,,\", $${num}}' $post_res_f)"
      done
      echo
     
    done

fi


