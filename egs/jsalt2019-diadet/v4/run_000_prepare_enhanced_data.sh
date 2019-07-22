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

if [ $stage -le 1 ];then
    # Prepare the enhanced data of Babytrain.
    for partition in {train,dev,test}
    do
      dataDir=/export/fs01/jsalt19/databases/BabyTrain/${partition}/wav
      outputDir=${OUTPUT_ROOT}/BabyTrain/${partition}
      echo "$train_cmd_gpu  ./local/denoising_sig2sig.sh $dataDir $outputDir "

      $se_cmd_gpu ./local/denoising_sig2sig.sh $dataDir $outputDir  
      exit
    done

fi
