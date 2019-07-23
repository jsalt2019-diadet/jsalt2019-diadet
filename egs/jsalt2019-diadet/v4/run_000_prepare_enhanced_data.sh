#!/bin/bash
# This script demonstrates how to run speech enhancement.


###################################
# Run speech enhancement
###################################
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

output_root=data_enhanced_audios
log_dir=$output_root/log/

echo "Processing all datasets can be very time-consuming, it's recommended to use the pre-processed audios path introduced in datapath.sh"
exit


if [ $stage -le 1 ];then
    # Prepare the enhanced data of Babytrain.
    for partition in {train,dev,test}
    do
      dataDir=/export/fs01/jsalt19/databases/BabyTrain/${partition}/wav
      outputDir=${output_root}/BabyTrain/${partition}
      (
         $se_cmd_gpu $log_dir/BabyTrian_log_${partition}.log \
            local/denoising_sig2sig.sh $dataDir $outputDir || exit 1;
      ) &
      sleep 15
       
    done

fi

if [ $stage -le 2 ];then
    # Prepare the enhanced data of AMI.
    for partition in {train,dev,test}
    do
      dataDir=/export/fs01/jsalt19/databases/AMI/${partition}/wav/
      outputDir=${output_root}/AMI/${partition}

      (
         $se_cmd_gpu $log_dir/AMI_log_${partition}.log \
            local/denoising_sig2sig.sh $dataDir $outputDir || exit 1;
      ) &
      sleep 15   
    
    done

fi

if [ $stage -le 3 ];then
    # Prepare the enhanced data of chime-5.
    for partition in {train,dev,test}
    do
      dataDir=/export/fs01/jsalt19/databases/CHiME5/${partition}/wav/
      outputDir=${output_root}/CHiME5/${partition}

      (
         $se_cmd_gpu $log_dir/chime5_log_${partition}.log \
            local/denoising_sig2sig.sh $dataDir $outputDir || exit 1;
      ) &
      sleep 15
      
    done

fi

if [ $stage -le 4];then
    # Prepare the enhanced data of sri.
    for partition in {train,dev,test}
    do
      dataDir=/export/fs01/jsalt19/databases/SRI/${partition}/wav/
      outputDir=${output_root}/SRI/${partition}
      
      (
         $se_cmd_gpu $log_dir/sri_log_${partition}.log \
            local/denoising_sig2sig.sh $dataDir $outputDir || exit 1;
      ) &
      sleep 15
      
    done

fi










