#!/bin/bash

# this script expects pyannote-audio to be installed like follows, until I make a
# proper pyannote-audio 2.0 release. DO NOT USE pyannote-audio 1.0 FROM PyPI
# $ git clone https://github.com/pyannote/pyannote-audio.git
# $ cd pyannote-audio
# $ git checkout develop   <--- important: one must used the develop branch
# $ pip install .


. ./cmd.sh
. ./path.sh
set -e
stage=1

. parse_options.sh || exit 1;

vaddir_supervad=`pwd`/vad_supervad  


ami_prots=(AMI.SpeakerDiarization.{MixHeadset,Array1,Array2})
babytrain_prots=(BabyTrain.SpeakerDiarization.All)
chime5_prots=(CHiME5.SpeakerDiarization.{U01,U06})
sri_prots=(SRI.SpeakerDiarization.All)
all_prots="${ami_prots[@]} ${babytrain_prots[@]} ${chime5_prots[@]} ${sri_prots[@]}"

# TESTING(FIXME) - do we need to include SRI ?
all_prots="${ami_prots[@]} ${babytrain_prots[@]} ${chime5_prots[@]}"



if [ $stage -le 1 ];then 

  for PROTOCOL in  $all_prots           
  do

    $pyannote_cmd --gpu 1 exp/vad/${PROTOCOL}/apply.log \
      steps_vad/apply.sh ${PROTOCOL} &
      
  done

fi 
wait


if [ $stage -le 2 ];then 
  # take Pyannote rttm, copy to it's respective kaldi-style data dir 
  # also make a binary vad for the data dir, replacing the evad vad.scp
  # propogate vad.scp to enhancement dirs too

  for PROTOCOL in  $all_prots           
  do

    # models are stored here
    EXPERIMENT_DIR="exp/vad/${PROTOCOL}"  

    declare -A mapping=( ["AMI.SpeakerDiarization.Array1"]="jsalt19_spkdiar_ami_dev_Array1-01" \
                        ["AMI.SpeakerDiarization.Array2"]="jsalt19_spkdiar_ami_dev_Array2-01" \
                        ["AMI.SpeakerDiarization.MixHeadset"]="jsalt19_spkdiar_ami_dev_Mix-Headset" \
                        ["BabyTrain.SpeakerDiarization.All"]="jsalt19_spkdiar_babytrain_dev" \
                        ["CHiME5.SpeakerDiarization.U01"]="jsalt19_spkdiar_chime5_dev_U01" \
                        ["CHiME5.SpeakerDiarization.U06"]="jsalt19_spkdiar_chime5_dev_U06" \
                        ["SRI.SpeakerDiarization.All"]="jsalt19_spkdiar_sri_dev" )
    declare -A mapping_enhanced=( ["AMI.SpeakerDiarization.Array1"]="jsalt19_spkdiar_ami_enhanced_dev_Array1-01" \
                        ["AMI.SpeakerDiarization.Array2"]="jsalt19_spkdiar_ami_enhanced_dev_Array2-01" \
                        ["AMI.SpeakerDiarization.MixHeadset"]="jsalt19_spkdiar_ami_enhanced_dev_Mix-Headset" \
                        ["BabyTrain.SpeakerDiarization.All"]="jsalt19_spkdiar_babytrain_enhanced_dev" \
                        ["CHiME5.SpeakerDiarization.U01"]="jsalt19_spkdiar_chime5_enhanced_dev_U01" \
                        ["CHiME5.SpeakerDiarization.U06"]="jsalt19_spkdiar_chime5_enhanced_dev_U06" \
                        ["SRI.SpeakerDiarization.All"]="jsalt19_spkdiar_sri_enhanced_dev" ) 
    cp ${EXPERIMENT_DIR}/results/${PROTOCOL}.development.rttm ./data/${mapping[$PROTOCOL]}/pyannote_vad.rttm
    cp ${EXPERIMENT_DIR}/results/${PROTOCOL}.development.rttm ./data/${mapping_enhanced[$PROTOCOL]}/pyannote_vad.rttm

    # Usage: rttm_to_bin_vad.sh [options] <rttm-file> <data-dir> <path-to-vad-dir>
    # overwrite evad vad.scp in the data dir 
    num_utt=$(wc -l data/${mapping[$PROTOCOL]}/utt2spk | cut -d " " -f 1)
    nj=$(($num_utt < 5 ? 1:5))
    mv data/${mapping[$PROTOCOL]}/vad.scp data/${mapping[$PROTOCOL]}/evad.scp
    hyp_utils/rttm_to_bin_vad.sh --nj $nj data/${mapping[$PROTOCOL]}/pyannote_vad.rttm data/${mapping[$PROTOCOL]} $vaddir_supervad
    mv data/${mapping[$PROTOCOL]}/vad.scp data/${mapping[$PROTOCOL]}/pyannote_vad.scp
    ln -rs data/${mapping[$PROTOCOL]}/pyannote_vad.scp data/${mapping[$PROTOCOL]}/vad.scp

    # take vad.scp from the unenhanced data dir, copy over to enhanced
    mv data/${mapping_enhanced[$PROTOCOL]}/vad.scp data/${mapping_enhanced[$PROTOCOL]}/evad.scp
    cp data/${mapping[$PROTOCOL]}/pyannote_vad.scp data/${mapping_enhanced[$PROTOCOL]}/pyannote_vad.scp
    ln -rs data/${mapping_enhanced[$PROTOCOL]}/pyannote_vad.scp data/${mapping_enhanced[$PROTOCOL]}/vad.scp

    utils/fix_data_dir.sh data/${mapping[$PROTOCOL]}
    utils/fix_data_dir.sh data/${mapping_enhanced[$PROTOCOL]}




    declare -A mapping=( ["AMI.SpeakerDiarization.Array1"]="jsalt19_spkdiar_ami_eval_Array1-01" \
                        ["AMI.SpeakerDiarization.Array2"]="jsalt19_spkdiar_ami_eval_Array2-01" \
                        ["AMI.SpeakerDiarization.MixHeadset"]="jsalt19_spkdiar_ami_eval_Mix-Headset" \
                        ["BabyTrain.SpeakerDiarization.All"]="jsalt19_spkdiar_babytrain_eval" \
                        ["CHiME5.SpeakerDiarization.U01"]="jsalt19_spkdiar_chime5_eval_U01" \
                        ["CHiME5.SpeakerDiarization.U06"]="jsalt19_spkdiar_chime5_eval_U06" \
                        ["SRI.SpeakerDiarization.All"]="jsalt19_spkdiar_sri_eval" )
    declare -A mapping_enhanced=( ["AMI.SpeakerDiarization.Array1"]="jsalt19_spkdiar_ami_enhanced_eval_Array1-01" \
                        ["AMI.SpeakerDiarization.Array2"]="jsalt19_spkdiar_ami_enhanced_eval_Array2-01" \
                        ["AMI.SpeakerDiarization.MixHeadset"]="jsalt19_spkdiar_ami_enhanced_eval_Mix-Headset" \
                        ["BabyTrain.SpeakerDiarization.All"]="jsalt19_spkdiar_babytrain_enhanced_eval" \
                        ["CHiME5.SpeakerDiarization.U01"]="jsalt19_spkdiar_chime5_enhanced_eval_U01" \
                        ["CHiME5.SpeakerDiarization.U06"]="jsalt19_spkdiar_chime5_enhanced_eval_U06" \
                        ["SRI.SpeakerDiarization.All"]="jsalt19_spkdiar_sri_enhanced_eval" )                        
    cp ${EXPERIMENT_DIR}/results/${PROTOCOL}.test.rttm ./data/${mapping[$PROTOCOL]}/pyannote_vad.rttm
    cp ${EXPERIMENT_DIR}/results/${PROTOCOL}.test.rttm ./data/${mapping_enhanced[$PROTOCOL]}/pyannote_vad.rttm

    # Usage: rttm_to_bin_vad.sh [options] <rttm-file> <data-dir> <path-to-vad-dir>
    # overwrite evad vad.scp in the data dir 
    num_utt=$(wc -l data/${mapping[$PROTOCOL]}/utt2spk | cut -d " " -f 1)
    nj=$(($num_utt < 5 ? 1:5))
    mv data/${mapping[$PROTOCOL]}/vad.scp data/${mapping[$PROTOCOL]}/evad.scp
    hyp_utils/rttm_to_bin_vad.sh --nj $nj data/${mapping[$PROTOCOL]}/pyannote_vad.rttm data/${mapping[$PROTOCOL]} $vaddir_supervad
    mv data/${mapping[$PROTOCOL]}/vad.scp data/${mapping[$PROTOCOL]}/pyannote_vad.scp
    ln -rs data/${mapping[$PROTOCOL]}/pyannote_vad.scp data/${mapping[$PROTOCOL]}/vad.scp

    # take vad.scp from the unenhanced data dir, copy over to enhanced
    mv data/${mapping_enhanced[$PROTOCOL]}/vad.scp data/${mapping_enhanced[$PROTOCOL]}/evad.scp
    cp data/${mapping[$PROTOCOL]}/pyannote_vad.scp data/${mapping_enhanced[$PROTOCOL]}/pyannote_vad.scp
    ln -rs data/${mapping_enhanced[$PROTOCOL]}/pyannote_vad.scp data/${mapping_enhanced[$PROTOCOL]}/vad.scp

    utils/fix_data_dir.sh data/${mapping[$PROTOCOL]}
    utils/fix_data_dir.sh data/${mapping_enhanced[$PROTOCOL]}

  done

fi


