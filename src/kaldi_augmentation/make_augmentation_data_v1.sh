#!/bin/bash

. ./cmd.sh
. ./path.sh

set -e

stage=0

sampling_rate=16000

train_dir_list="train_subset" # List of train dirs that you want to augment
eval_dir_list="sitw_eval_test"   # List of eval dirs that you want to augment
train_snrs="15 10 5 0"  # snr ranges that you want to augment the train dirs
eval_snrs="17 12 7 2" # snr ranges that you want to augment the test dirs

# Reverberation opts
sim_rirs_path=RIRS_NOISES/simulated_rirs
rt60_info_path=/home/snidada1/usr/src/workspace/workspace/pytorch_utils/far_field_simulation/rt60_for_simulated_rirs
sim_rirs_src_location=/export/b17/snidada1/kaldi_jsalt_2019/egs/voxceleb/v2/RIRS_NOISES

if [ $stage -le 0 ]; then
  # First make the MUSAN corpus
  # We will make 90-10 splits of speech, noise and music directories
  # The 90 split will be used for augmenting the train directories
  # The 10 split will be used for augmenting the eval directories
  steps/data/make_musan.sh --sampling-rate $sampling_rate /export/corpora/JHU/musan data
  for name in speech noise music; do
    utils/subset_data_dir_tr_cv.sh data/musan_${name} \
            data/musan_${name}_train data/musan_${name}_eval
  done

  for name in speech noise music; do
    for mode in train eval; do
      utils/data/get_utt2dur.sh data/musan_${name}_${mode}
      mv data/musan_${name}_${mode}/utt2dur data/musan_${name}_${mode}/reco2dur
    done
  done
fi


if [ $stage -le 1 ]; then
  # Augment the train directories
  for name in $train_dir_list; do
    utils/data/get_reco2dur.sh data/$name
    for snr in $train_snrs; do
      # Augment with musan_noise
      steps/data/augment_data_dir.py --utt-suffix "noise-snr$snr" --fg-interval 1 \
            --fg-snrs "$snr" --fg-noise-dir "data/musan_noise_train" \
            --modify-spk-id "false" \
            data/$name data/${name}_noise_snr$snr
      awk -v snr=$snr '{print $0" "snr}' data/${name}_noise_snr$snr/utt2uniq | sort -k1,1 > data/${name}_noise_snr$snr/utt2snr
      utils/fix_data_dir.sh --utt-extra-files "utt2snr" data/${name}_noise_snr$snr
      # Augment with musan_music
      steps/data/augment_data_dir.py --utt-suffix "music-snr$snr" --bg-snrs "$snr" \
            --num-bg-noises "1" --bg-noise-dir "data/musan_music_train" \
            --modify-spk-id "false" \
            data/$name data/${name}_music_snr$snr
      awk -v snr=$snr '{print $0" "snr}' data/${name}_music_snr$snr/utt2uniq | sort -k1,1 > data/${name}_music_snr$snr/utt2snr
      utils/fix_data_dir.sh --utt-extra-files "utt2snr" data/${name}_music_snr$snr
      # Augment with musan_speech
      steps/data/augment_data_dir.py --utt-suffix "babble-snr$snr" --bg-snrs "$snr" \
            --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech_train" \
            --modify-spk-id "false" \
            data/$name data/${name}_babble_snr$snr
      awk -v snr=$snr '{print $0" "snr}' data/${name}_babble_snr$snr/utt2uniq | sort -k1,1 > data/${name}_babble_snr$snr/utt2snr
      utils/fix_data_dir.sh --utt-extra-files "utt2snr" data/${name}_babble_snr$snr
    done

    # Also combine different snr directories to a single directory
    for aug_type in babble noise music; do
      combine_str=""
      for snr in $train_snrs; do
        combine_str=$combine_str" data/${name}_${aug_type}_snr$snr"
      done
      utils/combine_data.sh --extra-files "utt2snr" data/${name}_${aug_type} $combine_str
    done
  done
fi

if [ $stage -le 2 ]; then
  # Augment the eval directories
  for name in $eval_dir_list; do
    utils/data/get_reco2dur.sh data/$name
    for snr in $eval_snrs; do
      # Augment with musan_noise
      steps/data/augment_data_dir.py --utt-suffix "noise-snr$snr" --fg-interval 1 \
            --fg-snrs "$snr" --fg-noise-dir "data/musan_noise_eval" \
            data/$name data/${name}_noise_snr$snr
      awk -v snr=$snr '{print $0" "snr}' data/${name}_noise_snr$snr/utt2uniq | sort -k1,1 > data/${name}_noise_snr$snr/utt2snr
      utils/fix_data_dir.sh --utt-extra-files "utt2snr" data/${name}_noise_snr$snr
      # Augment with musan_music
      steps/data/augment_data_dir.py --utt-suffix "music-snr$snr" --bg-snrs "$snr" \
            --num-bg-noises "1" --bg-noise-dir "data/musan_music_eval" \
            data/$name data/${name}_music_snr$snr
      awk -v snr=$snr '{print $0" "snr}' data/${name}_music_snr$snr/utt2uniq | sort -k1,1 > data/${name}_music_snr$snr/utt2snr
      utils/fix_data_dir.sh --utt-extra-files "utt2snr" data/${name}_music_snr$snr
      # Augment with musan_speech
      steps/data/augment_data_dir.py --utt-suffix "babble-snr$snr" --bg-snrs "$snr" \
            --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech_eval" \
            data/$name data/${name}_babble_snr$snr
      awk -v snr=$snr '{print $0" "snr}' data/${name}_babble_snr$snr/utt2uniq | sort -k1,1 > data/${name}_babble_snr$snr/utt2snr
      utils/fix_data_dir.sh --utt-extra-files "utt2snr" data/${name}_babble_snr$snr
    done
  done
fi

# Reverberant speech simulation
if [ $stage -le 3 ]; then
  # We will filter the room impulse responses into three directories bases on rt60s

  # downloading everytime is a time taking and disk consuming process
  # It is better to softlink from a location that we will not delete untill the end of workshop
  if [ ! -d "RIRS_NOISES" ]; then
    ln -s $sim_rirs_src_location RIRS_NOISES
  fi

  #if [ ! -d "RIRS_NOISES" ]; then
  #  # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
  #  wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
  #  unzip rirs_noises.zip
  #fi

  cat $rt60_info_path/{smallroom,mediumroom,largeroom}_rt60s.txt | sort -k1,1 > $sim_rirs_path/simrirs2rt60.map

  cat $sim_rirs_path/{smallroom,mediumroom,largeroom}/rir_list > $sim_rirs_path/rir_list_all_rooms

  # Split the rooms into two lists (train and test). There are total of 600 rooms split them into 90-10 list (540-60)
  utils/shuffle_list.pl $sim_rirs_path/simrirs2rt60.map | head -540 | sort -k1,1 > $sim_rirs_path/simrirs2rt60_train.map
  utils/filter_scp.pl --exclude $sim_rirs_path/simrirs2rt60_train.map $sim_rirs_path/simrirs2rt60.map | sort -k1,1 > $sim_rirs_path/simrirs2rt60_eval.map

  # First filter out the rirs based on the value of rt60s
  for mode in train eval; do
    # list for 0.0 < rt60 < 0.2
    awk '$2 < 0.2 {print $1}' $sim_rirs_path/simrirs2rt60_${mode}.map | \
      utils/filter_scp.pl -f 4 - $sim_rirs_path/rir_list_all_rooms > $sim_rirs_path/rir_list_${mode}_rt60_min_0.0_max_0.2

    # list for 0.2 < rt60 < 0.6
    awk '$2 >= 0.2 && $2 < 0.6 {print $1}' $sim_rirs_path/simrirs2rt60_${mode}.map | \
      utils/filter_scp.pl -f 4 - $sim_rirs_path/rir_list_all_rooms > $sim_rirs_path/rir_list_${mode}_rt60_min_0.2_max_0.6

    # list for 0.6 < rt60 < inf
    awk '$2 > 0.6 {print $1}' $sim_rirs_path/simrirs2rt60_${mode}.map | \
      utils/filter_scp.pl -f 4 - $sim_rirs_path/rir_list_all_rooms > $sim_rirs_path/rir_list_${mode}_rt60_min_0.6_max_2.0
  done
fi

if [ $stage -le 4 ]; then
  # Reverberate speech using RIRs in the range 0.0 < rt60 < 0.2 for train dirs
  mode=train
  for name in $train_dir_list; do
    for rt60_range in 0.0:0.2 0.2:0.6 0.6:2.0; do
      # Reverberate speech using RIRs in the range 0.0 < rt60 < 0.2
      rt60_min=`echo $rt60_range | cut -d ":" -f1`
      rt60_max=`echo $rt60_range | cut -d ":" -f2`
      kwrd=rt60_min_${rt60_min}_max_${rt60_max}

      # Make a version with reverberated speech
      rvb_opts=()
      rvb_opts+=(--rir-set-parameters "0.5, $sim_rirs_path/rir_list_${mode}_${kwrd}")

      # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
      # additive noise here.
      steps/data/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --speech-rvb-probability 1 \
        --pointsource-noise-addition-probability 0 \
        --isotropic-noise-addition-probability 0 \
        --num-replications 1 \
        --source-sampling-rate $sampling_rate \
        data/$name data/${name}_reverb_${kwrd}
      cp data/$name/vad.scp data/${name}_reverb_${kwrd}
      utils/copy_data_dir.sh --utt-suffix "-reverb-rt60-${rt60_min}-${rt60_max}" data/${name}_reverb_${kwrd} data/${name}_reverb_${kwrd}.new
      rm -rf data/${name}_reverb_${kwrd}
      mv data/${name}_reverb_${kwrd}.new data/${name}_reverb_${kwrd}

      # Create utt2rt60 file
      python local/make_utt2reverb_info.py data/${name}_reverb_${kwrd} $sim_rirs_path/simrirs2rt60.map \
            data/${name}_reverb_${kwrd}/utt2reverbinfo || exit 1;
   done
 done
fi

if [ $stage -le 5 ]; then
  # Simulate reverberant speech for eval dirs with 0.0 < rt60 < 0.2
  mode=eval
  for name in $eval_dir_list; do
    for rt60_range in 0.0:0.2 0.2:0.6 0.6:2.0; do
      rt60_min=`echo $rt60_range | cut -d ":" -f1`
      rt60_max=`echo $rt60_range | cut -d ":" -f2`
      kwrd=rt60_min_${rt60_min}_max_${rt60_max}

      # Make a version with reverberated speech
      rvb_opts=()
      rvb_opts+=(--rir-set-parameters "0.5, $sim_rirs_path/rir_list_${mode}_${kwrd}")

      # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
      # additive noise here.
      steps/data/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --speech-rvb-probability 1 \
        --pointsource-noise-addition-probability 0 \
        --isotropic-noise-addition-probability 0 \
        --num-replications 1 \
        --source-sampling-rate $sampling_rate \
        data/$name data/${name}_reverb_${kwrd}
      cp data/$name/vad.scp data/${name}_reverb_${kwrd}
      utils/copy_data_dir.sh --utt-suffix "-reverb-rt60-${rt60_min}-${rt60_max}" data/${name}_reverb_${kwrd} data/${name}_reverb_${kwrd}.new
      rm -rf data/${name}_reverb_${kwrd}
      mv data/${name}_reverb_${kwrd}.new data/${name}_reverb_${kwrd}

      # Create utt2rt60 file
      local/make_utt2reverb_info.py data/${name}_reverb_${kwrd} $sim_rirs_path/simrirs2rt60.map \
            data/${name}_reverb_${kwrd}/utt2reverbinfo || exit 1;
    done
  done
fi


# So far we created several copies of train and eval directories
# At this stage you will have several copies of augmentation directories created at individual SNRs
# Also several reverb directories based on rt60 values
# It is upto the user to decide on how to combine these directories to form one single training dir
# We provide an example below
if [ $stage -le 6 ]; then
  # Combine different reverb directories to a single directory
  # Example is given below
  for name in $train_dir_list; do
    combine_str=""
    for kwrd in rt60_min_0.0_max_0.2 rt60_min_0.2_max_0.6 rt60_min_0.6_max_2.0; do
      combine_str=$combine_str" data/${name}_reverb_${kwrd}"
      utils/combine_data.sh --extra-files "utt2reverbinfo" data/${name}_reverb $combine_str
    done
  done
fi

if [ $stage -le 7 ]; then
  # Combine different additive noise based augmentation directories
  for name in $train_dir_list; do
    combine_str=""
    for aug in babble music noise; do
      combine_str=$combine_str" data/${name}_${aug}"
      utils/combine_data.sh --extra-files "utt2snr" data/${name}_additive_aug $combine_str
    done
  done
fi

if [ $stage -le 8 ]; then
  # Finally combine both reverb and additive noise directories to form a single directory
  # A subset of this would be used for training
  # It is upto the user to decide how to make that subset
  # utils/subset_data_dir.sh --utt-list ${use_defined_utt_lisr} data/${name}_additive_aug data/${name}_subset_user
  for name in $train_dir_list; do
     utils/combine_data.sh data/${name}_reverb_additive \
                    data/${name}_additive_aug data/${name}_reverb || exit 0;
  done
fi

if [ $stage -le 9 ]; then
  # Make a single utt2info file that summarizes the all the utt info (rt60s, noise type, snrs and other info)
  for name in $train_dir_list; do
    local/make_utt2info.py --additive-noise-types music babble noise \
                        --utt2snr-file data/${name}_additive_aug/utt2snr \
                        --utt2reverb-file data/${name}_reverb/utt2reverbinfo \
                        --utt2info-file data/${name}_reverb_additive/utt2info
  done
fi
