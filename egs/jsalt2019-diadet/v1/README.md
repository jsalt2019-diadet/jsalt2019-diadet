# jsalt2019-diadet/v1

Recipe for speaker diarization detection and tracking for JSALT2019 workshop on
"Speaker Detection in Adverse Scenarios with a Single Microphone"

It will run for the workshop tasks based on datasets:
 - BabyTrain
 - Chime5
 - Ami

## How to run

The recipe has a style similar to Kaldi recipes. However, instead of having a unique run.sh bash script that runs all the steps, we divided the recipe in several scripts. Each script is named as run_XXX_*.sh where XXX a number which indicates its order in the sequence. We decided to split the recipe in several scripts because in most case you won't need to run the recipe from the beginning to the end. Someone can provide you with some precomputed features or pretrained neural networks and then you just need to run the last steps corresponding to the PLDA back-end.

The numbering of the scripts follows this convention:
 - 00X: data preparation and feature extraction
 - 01X: x-vector training
 - 02X: speaker diarization steps
 - 03X: x-vector extraction
 - 04X: PLDA back-ends for speaker detection/tracking.
 - 05X: print results

Please, read and understand what each script does before running it.

## Directory structure

The recipe contains the following directories:
 - ./: contains scripts for all the steps
 - ./conf: configuration files for SGE, feature extraction, etc.
 - ./data: kaldi stype data directory
 - ./exp: contains all data for the experiment
    - models: NNets, plda
    - xvectors
    - scores
    - results
    - etc
 - ./local: auxiliary scripts related to this recipe
     - scripts to create kaldi stype data directories for each dataset.
     - scripts to convert between formats
     - scripts to calibrate, compute dcf, der
 - steps/kaldi_steps: link to kaldi steps directory
 - utils/kaldi_utils: link to kaldi utils directory
 - hyp_utils: link to hyperion utils directory
 - steps_be: scripts for the steps of the PLDA back-end
 - steps_fe: scripts for some front-end tasks like kaldi VAD
 - steps_kaldi_diar: scripts for diarization using kaldi tools.
 - steps_kaldi_xvec: scripts to train and compute x-vectors using kaldi tools


## Resources

   There are precomputed networks, features, etc in this path of the CLSE grid
   ```bash
   /export/fs01/jsalt19/resources
   ```

   This is the baseline kaldi x-vector network trained with half of voxceleb without augmentations:
   ```bash
   /export/fs01/jsalt19/resources/embeddings_nnets/kaldi_xvec/mfcc40/xvector_nnet_2a.1.voxceleb_div2
   ```

## Auxiliary scripts

 - cmd.sh: define different commands to submit jobs to the SGE queue
 - path.sh: environment variables with the location of all the tools needed by the experiments: python, kaldi, hyperion, cuda, etc.
 - datapath.sh environment variables with the location of the datasets in the grid.


## Experiment configuration file

The default configuration parameters are defined in default_config.sh
In this file there are environment variables that define things like:
 - x-vector/plda/lda training data
 - score-normalization data
 - plda/lda dimensions
 - type of plda model
 - nnet directory name
 - back-end/score directory names
 - etc.

All the run_XXX_*.sh files will use default_config.sh
If you want to change some config parameters you can either:
 - Edit default_config.sh
 - Create a new config file, e.g., my_config.sh and call the recipe scripts as
 ```bash
 run_XXX_yyyyy.sh --config-file my_config.sh
 ```

## Recipe steps

This is a summary of the recipe steps:

 - run_001_prepare_data.sh:
      - Prepares all the training and evaluation data by createing kaldi style data directories with usual files: wav.scp, utt2spk, spk2utt, ...

 - run_002_compute_mfcc_evad.sh:
      - Computes MFCC and energy VAD for all datasets.
      - It also creates data directories with ground truth VAD for the evaluation data.

 - run_003_prepare_augment.sh:
      - Creates augmented data directory for voxceleb data.
      - It augments voxceleb with noise and reverberation.
      - This augmented data is used to train x-vector and LDA/PLDA.
      - However, if you use the default configuration you don't need to run this script since the default configuration only use non-augmented data to speed up the completion of the experiment.

 - run_004_compute_mfcc_augment.sh:
      - Computes MFCC for the augmented data and merges original and augmented data directories.
      - Again, if you use the default configuration you don't need to run this script.

 - run_010_prepare_xvec_train_data.sh:
      - Prepares the features for x-vector training, i.e., removes silence and applies CMN.

 - run_011_train_xvector.sh:
      - Trains kaldi x-vector nnet.

 - run_020_prepare_data_for_diar.sh:
      - Applies CMN to features for diarization
      - Creates segmented data directories from original data directories:
           - Each continous speech segment is assigned a new utt-id = original-utt-id-time-begin-time-end based on ground truth VAD in the rttm file or binary VAD from Energy VAD.
	   - The feat.scp file is modified to split the feature matrices into a matrix per subsegment

