#!/bin/bash
set -e

stage=0
sampling_rate=16000
snrs="15 10 5 0"
rirs_info_path=data/rirs_info
mode=train # Train or eval
suffix=_ps # ps stands for point source on top of reverb

echo "$0 $@"  # Print the command line for logging.

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -le 1 ]; then
  cat >&2 <<EOF
    echo USAGE: $0 [--optional-args] <list of all dirs to augment>
    echo USAGE: $0 --sampling-rate 16000 --snrs \'15 10 5 0\' --mode=train --rirs-info-path data/rirs_info train1 train2
    echo USAGE: $0 --sampling-rate 16000 --snrs \'17 12 7 2\' --rirs-info-path data/rirs_info sitw_eval_test sitw_eval_enroll
    optional-args:
        --sampling-rate <16000>  # Specify the source sampling rate, default:16000
        --snrs  <snr range>      # Specify snr range, defaults: "15 10 5 0"
        --mode <train/val>       # Specify whether the inp dirs are train or eval, default:train
        --rirs-info-path <dir containing the train, eval list partitions> # This will be created in step1
        --stage <int>
EOF
    exit 1;
fi

dir_list=$@

echo Applying augmentation to directories: $dir_list
echo SNR range is set to $snrs

# Reverberation opts
sim_rirs_path=RIRS_NOISES/simulated_rirs

# Check if the dirs exist
for d in $dir_list; do
    [ ! -d data/$d ] && echo dir data/$d does not exist && exit 1;
done

if [ $stage -le 0 ]; then
  # Reverberate speech using RIRs in the range 0.0 < rt60 < 0.2 for train dirs
  for name in $dir_list; do
    kwrds=""
    for rt60_range in 0.0:0.5 0.5:1.0 1.0:1.5 1.5:4.0; do
      # Reverberate speech using RIRs in the range 0.0 < rt60 < 0.2
      rt60_min=`echo $rt60_range | cut -d ":" -f1`
      rt60_max=`echo $rt60_range | cut -d ":" -f2`
      kwrd=rt60_min_${rt60_min}_max_${rt60_max}
      kwrds=" $kwrd"

      # Make a version with reverberated speech
      rvb_opts=()
      rvb_opts+=(--rir-set-parameters "0.5,$rirs_info_path/rir_list_${mode}_${kwrd}")
      # --noise-id 001 --noise-type isotropic --rir-id 00019 iso_noise.wav
      awk '{print "--noise-id "$1" --noise-type point-source "$2}' data/musan_noise_${mode}/wav.scp > $rirs_info_path/musan_noise_${mode}.lst
      rvb_opts+=(--noise-set-parameters "0.5,$rirs_info_path/musan_noise_${mode}.lst")

      # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
      # additive noise here.
      steps/data/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --speech-rvb-probability 1 \
        --pointsource-noise-addition-probability 1.0 \
        --isotropic-noise-addition-probability 0 \
        --num-replications 1 \
        --normalize-output false \
        --source-sampling-rate $sampling_rate \
        data/$name data/${name}_reverb_${kwrd}${suffix}
      cp data/$name/vad.scp data/${name}_reverb_${kwrd}${suffix}
      utils/copy_data_dir.sh --utt-suffix "-reverb-rt60-${rt60_min}-${rt60_max}" data/${name}_reverb_${kwrd}${suffix} data/${name}_reverb_${kwrd}${suffix}.new
      rm -rf data/${name}_reverb_${kwrd}${suffix}
      mv data/${name}_reverb_${kwrd}${suffix}.new data/${name}_reverb_${kwrd}${suffix}

      # Create utt2rt60 file
      python local/make_utt2reverb_info_ver2.py data/${name}_reverb_${kwrd}${suffix} $rirs_info_path/simrirs2rt60.map \
            data/${name}_reverb_${kwrd}${suffix}/utt2reverbinfo || exit 1;
      # TODO modify this section to add snrs and aug type
      # To create utt2info file for reverb only directories we are using an aribitrary snr value of 40db and aug type is set to None
      awk '{print $1" "$2" noise 40 "$3" "$4" "$5" "$6}' data/${name}_reverb_${kwrd}${suffix}/utt2reverbinfo > data/${name}_reverb_${kwrd}${suffix}/utt2info
   done
   combine_str=""
   for kwrd in $kwrds; do
      combine_str=$combine_str" data/${name}_reverb_${kwrd}${suffix}"
      utils/combine_data.sh --extra-files "utt2reverbinfo utt2info" data/${name}_reverb${suffix} $combine_str
    done
 done
fi
