#!/bin/bash

set -e

stage=0
sampling_rate=16000
download_rirs=false
seed=777
# SRC locations of musan and simulated rirs
musan_src_location=/export/corpora/JHU/musan
rirs_src_location=/export/b17/snidada1/kaldi_jsalt_2019/egs/voxceleb/v2/RIRS_NOISES # If download rirs is set to true this path will never be used
rt60_map_file=/home/snidada1/jsalt_2019/rir_info/simrir2rt60.info
echo "$0 $@"  # Print the command line for logging.

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
  cat >&2 <<EOF
    echo USAGE: $0 [--optional-args] rir_info_dir
    echo USAGE: $0 --sampling-rate 16000 data/rir_info
    optional-args:
        --sampling-rate <16000>  # Specify the source sampling rate, default:16000
        --download_rirs  <true/false>      # Specify whether to download rirs and make your own copy or soflink from a central location
        --seed <int>    # Default 777
        --stage <int>
EOF
    exit 1;
fi

# Directory to save rt60 information
#rirs_info_path=data/rirs_info
rirs_info_path=$1

if [ $stage -le 0 ]; then
  # First make the MUSAN corpus
  # We will make 90-10 splits of speech, noise and music directories
  # The 90 split will be used for augmenting the train directories
  # The 10 split will be used for augmenting the eval directories
  steps/data/make_musan.sh --sampling-rate $sampling_rate \
        $musan_src_location data
  for name in speech noise music; do
    utils/subset_data_dir_tr_cv.sh --seed $seed data/musan_${name} \
            data/musan_${name}_train data/musan_${name}_eval
  done

  for name in speech noise music; do
    for mode in train eval; do
      utils/data/get_utt2dur.sh data/musan_${name}_${mode}
      mv data/musan_${name}_${mode}/utt2dur data/musan_${name}_${mode}/reco2dur
    done
  done
  echo "Finished setting up MUSAN corpus"
fi

if [ $stage -le 1 ]; then
  # Make the demand and chime2 background noise dirs
  local/make_DEMAND_and_chime3background.py || exit 1;
  for name in demand_train chime3background_train chime3background_eval; do
    utils/fix_data_dir.sh data/$name
    utils/data/get_utt2dur.sh data/$name
    mv data/${name}/utt2dur data/${name}/reco2dur
    utils/fix_data_dir.sh data/$name
  done
fi

# Reverberant speech simulation
if [ $stage -le 2 ]; then
  if [ "$download_rirs" == true ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  else
    # downloading everytime is a time taking and disk consuming process
    # It is better to softlink from a location that we will not delete untill the end of workshop
    if [ ! -d "RIRS_NOISES" ]; then
        ln -s $sim_rirs_src_location RIRS_NOISES
    fi
  fi

  # The below script will compute rt60s based on sabine's formula for each of the roomtypes
  # We have three different room type's: smallroom. mediumroom and largeroom
  # It makes use of the room_info file to get the rooms dimensions and absorption coeff
  # It creates ${roomsize}_rt60s.txt in output dir RIRS_NOISES/simulated_rirs

  [ ! -d $rirs_info_path ] && mkdir -p $rirs_info_path
  cat RIRS_NOISES/simulated_rirs/{smallroom,mediumroom,largeroom}/rir_list > $rirs_info_path/rir_list.all

  # Split the rooms into two lists (train and test). There are total of 600 rooms split them into 90-10 list (540-60)
  awk '{print $2}' $rt60_map_file | sort -u > $rirs_info_path/room-ids.all
  utils/shuffle_list.pl --srand $seed $rirs_info_path/room-ids.all | \
        head -540 | sort -k1,1 > $rirs_info_path/room-ids.train
  utils/filter_scp.pl --exclude $rirs_info_path/room-ids.train \
        $rirs_info_path/room-ids.all \
        | sort -k1,1 > $rirs_info_path/room-ids.eval

  # Get the list of rirs for train and eval
  utils/filter_scp.pl -f 2 $rirs_info_path/room-ids.train $rt60_map_file \
        | awk '{print $1}' | sort -k1,1  \
        | utils/filter_scp.pl -f 2 - $rirs_info_path/rir_list.all \
        | sort -k1,1> $rirs_info_path/rir_list.train
  utils/filter_scp.pl -f 2 $rirs_info_path/room-ids.eval $rt60_map_file \
        | awk '{print $1}' | sort -k1,1  \
        | utils/filter_scp.pl -f 2 - $rirs_info_path/rir_list.all \
        | sort -k1,1> $rirs_info_path/rir_list.eval

  # First filter out the rirs based on the value of rt60s
  for mode in train eval; do
    # list for 0.0 < rt60 < 0.5
    awk '$3 < 0.5 {print $1}' $rt60_map_file | \
      utils/filter_scp.pl -f 2 - $rirs_info_path/rir_list.$mode > $rirs_info_path/rir_list_${mode}_rt60_min_0.0_max_0.5

    # list for 0.5 < rt60 < 1.0
    awk '$3 >= 0.5 && $3 < 1.0 {print $1}' $rt60_map_file | \
      utils/filter_scp.pl -f 2 - $rirs_info_path/rir_list.$mode > $rirs_info_path/rir_list_${mode}_rt60_min_0.5_max_1.0

    # list for 1.0 < rt60 < 1.5
    awk '$3 >= 1.0 && $3 < 1.5 {print $1}' $rt60_map_file | \
      utils/filter_scp.pl -f 2 - $rirs_info_path/rir_list.$mode > $rirs_info_path/rir_list_${mode}_rt60_min_1.0_max_1.5

    # list for 1.5 < rt60 < inf
    awk '$3 > 1.5 {print $1}' $rt60_map_file | \
      utils/filter_scp.pl -f 2 - $rirs_info_path/rir_list.$mode > $rirs_info_path/rir_list_${mode}_rt60_min_1.5_max_4.0
  done

  cat $rt60_map_file > $rirs_info_path/simrirs2rt60.map

  echo "Finished setting up the RIRs directory, lists are  saved in $rirs_info_path"
fi
