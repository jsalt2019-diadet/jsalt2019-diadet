#!/bin/bash
# Copyright 2018  Matthew Maciejewski
# Apache 2.0.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
num_components=2048
ivector_dim=128
stage=4
mic=sdm1
target_energy=0.3
mfcc_conf=conf/mfcc_dihard.conf
. utils/parse_options.sh

base_mic=$(echo $mic | sed 's/[0-9]//g') # sdm, ihm, or mdm
nmics=$(echo $mic | sed 's/[a-z]//g') # e.g. 8 for mdm8

AMI_DIR=$PWD/wav_db # Default,
case $(hostname -d) in
  fit.vutbr.cz) AMI_DIR=/mnt/matylda5/iveselyk/KALDI_AMI_WAV ;; # BUT,
  clsp.jhu.edu) AMI_DIR=/export/corpora4/ami/amicorpus ;; # JHU,
  cstr.ed.ac.uk) AMI_DIR= ;; # Edinburgh,
esac

# Prepare datasets
if [ $stage -le 0 ]; then
  if [ ! -f data/local/annotations/train.txt ]; then
    local/ami_text_prep.sh data/local/downloads
  fi
  local/ami_${base_mic}_data_prep.sh $AMI_DIR $mic
  echo "preparing data"
  local/ami_${base_mic}_scoring_data_prep.sh $AMI_DIR $mic dev
  local/ami_${base_mic}_scoring_data_prep.sh $AMI_DIR $mic eval

  python local/make_voxceleb_16khz.py /export/corpora/VoxCeleb data/voxceleb
  local/prepare_voxceleb_segments.sh /export/corpora/VoxCeleb data/voxceleb
  utils/fix_data_dir.sh data/voxceleb
fi

# Prepare features
if [ $stage -le 1 ]; then
#  steps/make_mfcc.sh --mfcc-config $mfcc_conf --nj 40 \
#    --cmd "$train_cmd" --write-utt2num-frames true \
#    data/voxceleb exp/make_mfcc $mfccdir
#  utils/fix_data_dir.sh data/voxceleb
#
#  for name in train_oraclespk dev eval; do
#    steps/make_mfcc.sh --mfcc-config $mfcc_conf --nj 40 \
#      --cmd "$train_cmd" --write-utt2num-frames true \
#      data/$mic/$name exp/$mic/make_mfcc $mfccdir
#    utils/fix_data_dir.sh data/$mic/$name
#  done
#
#  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
#    data/voxceleb exp/make_vad $vaddir
#  utils/fix_data_dir.sh data/voxceleb
#
#  for name in train_oraclespk dev eval; do
#    ns=$(cat data/$mic/$name/wav.scp | wc -l)
#    sid/compute_vad_decision.sh --nj $(($ns<40?$ns:40)) --cmd "$train_cmd" \
#      data/$mic/$name exp/$mic/make_vad $vaddir
#    utils/fix_data_dir.sh data/$mic/$name
#  done
#
#  utils/subset_data_dir.sh data/voxceleb 32000 data/voxceleb_32k

  utils/combine_data.sh data/$mic/train_voxceleb data/voxceleb data/$mic/train_oraclespk
  utils/subset_data_dir.sh data/$mic/train_voxceleb 32000 data/$mic/train_voxceleb_32k
fi

# Train UBM and i-vector extractor
if [ $stage -le 2 ]; then
  # Train UBM and i-vector extractor.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
    --nj 40 --num-threads 8 --delta-order 1 --apply-cmn false \
    data/$mic/train_voxceleb_32k $num_components exp/diag_ubm_$num_components

  sid/train_full_ubm.sh --nj 40 --remove-low-count-gaussians false \
    --cmd "$train_cmd --mem 25G" --apply-cmn false \
    data/$mic/train_voxceleb exp/diag_ubm_$num_components \
    exp/full_ubm_$num_components

  sid/train_ivector_extractor.sh \
    --cmd "$train_cmd --mem 35G" \
    --ivector-dim $ivector_dim --num-iters 5 --apply-cmn false \
    exp/full_ubm_$num_components/final.ubm data/$mic/train_voxceleb \
    exp/extractor_c${num_components}_i${ivector_dim}
fi

# Extract i-vectors
if [ $stage -le 3 ]; then
  # Extract iVectors for the training set, which is our PLDA training
  # data.  A long period is used here so that we don't compute too
  # many iVectors for each recording.
  diarization/extract_ivectors.sh --cmd "$train_cmd --mem 25G" \
    --nj 40 --window 1.5 --period 5.0 --min-segment 1.5 --apply-cmn false \
    --hard-min true exp/extractor_c${num_components}_i${ivector_dim} \
    data/$mic/train_voxceleb exp/$mic/ivectors_train_voxceleb

  diarization/extract_ivectors.sh --cmd "$train_cmd --mem 25G" \
    --nj 40 --window 1.5 --period 5.0 --min-segment 1.5 --apply-cmn false \
    --hard-min true exp/extractor_c${num_components}_i${ivector_dim} \
    data/$mic/train_oraclespk exp/$mic/ivectors_train_oraclespk

  # Extract iVectors for the two partitions of the test set.
  ns=$(cat data/$mic/dev/wav.scp | wc -l)
  diarization/extract_ivectors.sh --cmd "$train_cmd --mem 20G" \
    --nj $(($ns<40?$ns:40)) --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 exp/extractor_c${num_components}_i${ivector_dim} \
    data/$mic/dev exp/$mic/ivectors_dev

  ns=$(cat data/$mic/eval/wav.scp | wc -l)
  diarization/extract_ivectors.sh --cmd "$train_cmd --mem 20G" \
    --nj $(($ns<40?$ns:40)) --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 exp/extractor_c${num_components}_i${ivector_dim} \
    data/$mic/eval exp/$mic/ivectors_eval
fi

# Train PLDA models
if [ $stage -le 4 ]; then
  # Train a PLDA model on training set, using dev to whiten.
  # We will later use this to score iVectors in eval.
#  awk -F'[- ]' '{print $1"-"$2"-"$3" "$1}' exp/$mic/ivectors_train_oraclespk/utt2spk \
#    | utils/utt2spk_to_spk2utt.pl > exp/$mic/ivectors_train_oraclespk/seg2utt

  "$train_cmd" exp/$mic/ivectors_train_voxceleb/log/plda.log \
    ivector-compute-plda ark:exp/$mic/ivectors_train_voxceleb/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:exp/$mic/ivectors_train_voxceleb/ivector.scp ark:- \
      | transform-vec exp/$mic/ivectors_train_oraclespk/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    exp/$mic/ivectors_train_voxceleb/plda || exit 1;
fi

# Perform PLDA scoring
if [ $stage -le 5 ]; then
  # Perform PLDA scoring on all pairs of segments for each recording.
  ns=$(cat data/$mic/dev/wav.scp | wc -l)
  diarization/score_plda.sh --cmd "$train_cmd --mem 4G" --target-energy $target_energy \
    --nj $(($ns<20?$ns:20)) exp/$mic/ivectors_train_voxceleb exp/$mic/ivectors_dev \
    exp/$mic/ivectors_dev/plda_scores

  # Do the same thing for eval.
  ns=$(cat data/$mic/eval/wav.scp | wc -l)
  diarization/score_plda.sh --cmd "$train_cmd --mem 4G" --target-energy $target_energy \
    --nj $(($ns<20?$ns:20)) exp/$mic/ivectors_train_voxceleb exp/$mic/ivectors_eval \
    exp/$mic/ivectors_eval/plda_scores
fi

# Cluster the PLDA scores using a stopping threshold.
if [ $stage -le 6 ]; then
  # First, we find the threshold that minimizes the DER on each partition of
  # the test set.
  mkdir -p exp/$mic/tuning
  for dataset in dev eval; do
    ns=$(cat data/$mic/$dataset/wav.scp | wc -l)
    echo "Tuning clustering threshold for $dataset"
    best_der=100
    best_threshold=0

    # The threshold is in terms of the log likelihood ratio provided by the
    # PLDA scores.  In a perfectly calibrated system, the threshold is 0.
    # In the following loop, we evaluate the clustering on a heldout dataset
    # (dev is heldout for eval and vice-versa) using some reasonable
    # thresholds for a well-calibrated system.
    for threshold in -0.3 -0.2 -0.1 -0.05 0 0.05 0.1 0.2 0.3; do
      diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj $(($ns<20?$ns:20)) \
        --threshold $threshold exp/$mic/ivectors_$dataset/plda_scores \
        exp/$mic/ivectors_$dataset/plda_scores_t$threshold

      md-eval.pl -1 -c 0.25 -r data/$mic/$dataset/ref.rttm \
       -s exp/$mic/ivectors_$dataset/plda_scores_t$threshold/rttm \
       2> exp/$mic/tuning/${dataset}_t${threshold}.log \
       > exp/$mic/tuning/${dataset}_t${threshold}

      der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
        exp/$mic/tuning/${dataset}_t${threshold})
      if [ $(echo $der'<'$best_der | bc -l) -eq 1 ]; then
        best_der=$der
        best_threshold=$threshold
      fi
    done
    echo "$best_threshold" > exp/$mic/tuning/${dataset}_best
  done

  # Cluster dev using the best threshold found for eval.  This way,
  # eval is treated as a held-out dataset to discover a reasonable
  # stopping threshold for dev.
  ns=$(cat data/$mic/dev/wav.scp | wc -l)
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj $(($ns<20?$ns:20)) \
    --threshold $(cat exp/$mic/tuning/eval_best) \
    exp/$mic/ivectors_dev/plda_scores exp/$mic/ivectors_dev/plda_scores

  # Do the same thing for eval, treating dev as a held-out dataset
  # to discover a stopping threshold.
  ns=$(cat data/$mic/eval/wav.scp | wc -l)
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj $(($ns<20?$ns:20)) \
    --threshold $(cat exp/$mic/tuning/dev_best) \
    exp/$mic/ivectors_eval/plda_scores exp/$mic/ivectors_eval/plda_scores

  mkdir -p exp/$mic/results
  # Now combine the results for dev and eval and evaluate it
  # together.
  cat data/$mic/{dev,eval}/ref.rttm > data/local/$mic/fullref.rttm
  cat exp/$mic/ivectors_dev/plda_scores/rttm \
    exp/$mic/ivectors_eval/plda_scores/rttm | md-eval.pl -1 -c 0.25 -r \
    data/local/$mic/fullref.rttm -s - 2> exp/$mic/results/threshold.log \
    > exp/$mic/results/DER_threshold.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    exp/$mic/results/DER_threshold.txt)
  # Using supervised calibration, DER: #####%
  echo "Using supervised calibration, DER: $der%"
  echo "Target energy: $target_energy   DER: $der%" >> orig_mfcc_tuning_threshold.txt
fi

# Cluster the PLDA scores using the oracle number of speakers
if [ $stage -le 7 ]; then
  # In this section, we show how to do the clustering if the number of speakers
  # (and therefore, the number of clusters) per recording is known in advance.
  ns=$(cat data/$mic/dev/wav.scp | wc -l)
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj $(($ns<20?$ns:20)) \
    --reco2num-spk data/$mic/dev/reco2num_spk \
    exp/$mic/ivectors_dev/plda_scores exp/$mic/ivectors_dev/plda_scores_num_spk

  ns=$(cat data/$mic/eval/wav.scp | wc -l)
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj $(($ns<20?$ns:20)) \
    --reco2num-spk data/$mic/eval/reco2num_spk \
    exp/$mic/ivectors_eval/plda_scores exp/$mic/ivectors_eval/plda_scores_num_spk

  mkdir -p exp/$mic/results
  # Now combine the results for callhome1 and callhome2 and evaluate it together.
  cat exp/$mic/ivectors_dev/plda_scores_num_spk/rttm \
  exp/$mic/ivectors_eval/plda_scores_num_spk/rttm \
    | md-eval.pl -1 -c 0.25 -r data/local/$mic/fullref.rttm -s - 2> exp/$mic/results/num_spk.log \
    > exp/$mic/results/DER_num_spk.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    exp/$mic/results/DER_num_spk.txt)
  # Using the oracle number of speakers, DER: #####%
  echo "Using the oracle number of speakers, DER: $der%"
  echo "Target energy: $target_energy   DER: $der%" >> orig_mfcc_tuning_oraclespk.txt
fi
