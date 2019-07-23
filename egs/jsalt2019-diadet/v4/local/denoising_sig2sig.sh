#!/bin/bash
# speech enhancement: sig2sig

echo "Do speech denoising"
if [ $# -ne 2 ]; then
  echo "Usage: $0 <original-wav-root> <denoised-wav-root>"
  exit 1;
fi

set -e

data_dir=$1
output_dir=$2

# General settings
USE_GPU=true  # Use GPU instead of CPU. To instead use CPU, set to 'false'.
TRUNCATE_MINUTES=10  # Duration in minutes of chunks for enhancement. If you experience
                     # OOM errors with your GPU, try reducing this.
MODE=3   #Use which output of the model: mode=1 is irm, mode=2 is lps, mode=3 is fusion of both.
STAGE_SELECT=3 # Only works if choosing  1000h-model.

echo "/home/leisun1/anaconda3/bin/python  speech_denoising/main_denoising.py \
       --verbose \
       --wav_dir $data_dir --output_dir $output_dir \
       --use_gpu $USE_GPU  \
       --truncate_minutes $TRUNCATE_MINUTES \
       --mode $MODE \
       --stage_select $STAGE_SELECT"

/home/leisun1/anaconda3/bin/python  speech_denoising/main_denoising.py \
       --verbose \
       --wav_dir $data_dir --output_dir $output_dir \
       --use_gpu $USE_GPU  \
       --truncate_minutes  $TRUNCATE_MINUTES \
       --mode $MODE \
       --stage_select $STAGE_SELECT
