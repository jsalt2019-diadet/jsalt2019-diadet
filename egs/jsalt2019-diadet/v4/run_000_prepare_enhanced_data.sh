#!/bin/bash
# This script demonstrates how to run speech enhancement.


###################################
# Run speech enhancement
###################################
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. datapath.sh

OUTPUT_ROOT=./data_enhanced_aduios/ 
echo "Processing all datasets can be very time-consuming, it's recommended to use the pre-processed audios path introduced in datapath.sh"
exit


if [ $stage -le 1 ];then
    # Prepare the enhanced data of Babytrain.
    for partition in {train,dev,test}
    do
      dataDir=/export/fs01/jsalt19/databases/BabyTrain/${partition}/wav
      outputDir=${OUTPUT_ROOT}/BabyTrain/${partition}
      echo "$train_cmd_gpu  ./local/denoising_sig2sig.sh $dataDir $outputDir "

      $se_cmd_gpu ./local/denoising_sig2sig.sh $dataDir $outputDir  
    done

fi

if [ $stage -le 2 ];then
    # Prepare the enhanced data of AMI.
    for partition in {train,dev,test}
    do
      dataDir=/export/fs01/jsalt19/databases/AMI/${partition}/wav/
      outputDir=${OUTPUT_ROOT}/AMI/${partition}
      echo "$train_cmd_gpu  ./local/denoising_sig2sig.sh $dataDir $outputDir "

      $se_cmd_gpu ./local/denoising_sig2sig.sh $dataDir $outputDir  
    done

fi

if [ $stage -le 3 ];then
    # Prepare the enhanced data of chime-5.
    for partition in {train,dev,test}
    do
      dataDir=/export/fs01/jsalt19/databases/CHiME5/${partition}/wav/
      outputDir=${OUTPUT_ROOT}/CHiME5/${partition}
      echo "$train_cmd_gpu  ./local/denoising_sig2sig.sh $dataDir $outputDir "

      $se_cmd_gpu ./local/denoising_sig2sig.sh $dataDir $outputDir  
    done

fi

if [ $stage -le 4];then
    # Prepare the enhanced data of sri.
    for partition in {train,dev,test}
    do
      dataDir=/export/fs01/jsalt19/databases/SRI/${partition}/wav/
      outputDir=${OUTPUT_ROOT}/SRI/${partition}
      echo "$train_cmd_gpu  ./local/denoising_sig2sig.sh $dataDir $outputDir "

      $se_cmd_gpu ./local/denoising_sig2sig.sh $dataDir $outputDir  
    done

fi










