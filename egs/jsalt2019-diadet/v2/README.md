# jsalt2019-diadet/v2

Recipe for speaker diarization detection and tracking for JSALT2019 workshop on
"Speaker Detection in Adverse Scenarios with a Single Microphone"

It will run for the workshop tasks based on datasets:
 - BabyTrain
 - Chime5
 - Ami
 - SRI

## How to run

The recipe has a style similar to Kaldi recipes. However, instead of having a unique run.sh bash script that runs all the steps, we divided the recipe in several scripts. Each script is named as run_XXX_*.sh where XXX a number which indicates its order in the sequence. We decided to split the recipe in several scripts because in most case you won't need to run the recipe from the beginning to the end. Someone can provide you with some precomputed features or pretrained neural networks and then you just need to run the last steps corresponding to the PLDA back-end.

The numbering of the scripts follows this convention:
 - 00X: data preparation and feature extraction
 - 01X: x-vector training
 - 02X: speaker diarization steps
 - 03X: x-vector extraction for spk detection and tracking
 - 04X: PLDA back-ends for speaker detection
 - 05X: PLDA back-ends for speaker tracking
 - 06X: print results

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

   There are precomputed networks, features, etc in this path of the CLSP grid
   ```bash
   /export/fs01/jsalt19/resources
   ```

   This is the baseline kaldi x-vector network trained with half of voxceleb without augmentations:
   ```bash
   /export/fs01/jsalt19/resources/embeddings_nnets/kaldi_xvec/mfcc40/xvector_nnet_2a.1.voxceleb_div2
   ```

   This is the baseline kaldi x-vector network trained with voxceleb plus x2 augmentations with augmentations created in step run_003_prepare_augment.sh:
   ```bash
   /export/fs01/jsalt19/resources/embeddings_nnets/kaldi_xvec/mfcc40/xvector_nnet_2a.1.voxceleb_combined
   ```

   This is the data directory and MFCC40 precomputed features for the baseline
   ```bash
   /export/fs01/jsalt19/resources/feats/jsalt19-diadet/baseline_mfcc40/data
   ```
   You can copy this data directory to your egs/jsalt2019-diadet/v1 and skip steps 001 to 004

   These are precomputed xvectors to be used as input to the AHC diarization
   ```bash
   /export/fs01/jsalt19/resources/xvectors/jsalt19-diadet/baseline_mfcc40/xvectors_diar
   ```

   These are precomputed xvectors to be used as input to speaker detection back-end
   ```bash
   /export/fs01/jsalt19/resources/xvectors/jsalt19-diadet/baseline_mfcc40/xvectors
   ```

   The enhanced test sets (including dev, eval) of each corpus are precomputed and directly linked in testing scripts:
   ```
   /export/fs01/jsalt19/leisun/dataset/{BabyTrain,CHiME5,SRI,ami}/{dev,test}/SE_1000h_model_m3_s3/*.wav
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

- run_005_train_pyannote_vad.sh:
     - Trains LSTM-based VAD models on each dataset/microphone
     - Ensure that you have the right pyannote dependencies - (see script description)

- run_006_apply_pyannote_vad.sh:
     - Applies LSTM-based VAD on each dataset/microphone
     - Ensure that you have the right pyannote dependencies - (see script description)

 - run_010_prepare_xvec_train_data.sh:
      - Prepares the features for x-vector training, i.e., removes silence and applies CMN.

 - run_011_train_xvector.sh:
      - Trains kaldi x-vector nnet.

 - run_020_prepare_data_for_diar.sh:
      - Applies CMN to features for diarization
      - Creates segmented data directories from original data directories:
          - Each continous speech segment is assigned a new utt-id = original-utt-id-time-begin-time-end based on ground truth VAD in the rttm file or binary VAD from Energy VAD.
	  - The feat.scp file is modified to split the feature matrices into a matrix per subsegment

 - run_021_extract_xvectors_for_diar.sh
     - Compute xvectors with sliding window for:      
         - Dev and eval speaker diarization datasets
	 - Test part of dev and eval speaker detection/tracking datasets to cluster files into single speaker clusters.
	 - Voxceleb to train PLDA for diarization
	 - Train speaker diarization datsets to adapt PLDA

 - run_022_train_diar_be.sh
     - Trains diarization back-end (LDA, PLDA) using kaldi tools
        - Trains out-of-domain PLDA on voxceleb
        - Trains mixed-domain PLDAs on Voxceleb + train-part of each test dataset

 - run_023_eval_diar_be.sh
     - Evaluates AHC using out-of-domain PLDA
         - Optains optimum AHC threshold from dev part
	 - Results are left in, for example:
	 ```bash
	 exp/diarization/2a.1.voxceleb_div2/lda120_plda_voxceleb/jsalt19_spkdet_babytrain_eval_test_gtvad/plda_scores_tbest/result.md-eval
	 ```

 - run_024_eval_diar_be_adapt.sh
    - Evaluates AHC using mixed-domain PLDA
        - Results are left in, for example,
	```bash
	exp/diarization/2a.1.voxceleb_div2/lda120_plda_voxceleb_babytrain/jsalt19_spkdiar_babytrain_eval_gtvad/plda_scores_tbest/result.md-eval
	```
	
 - run_030_extract_xvectors_wo_diar.sh
    - Extracts x-vectors for spk detection without any diarization
       - Voxceleb for training back-end with energy VAD
       - Adaptation datasets for babytrain, chime5, ami with ground truth VAD
       - Enrollment datasets for dev/eval with ground truth VAD
       - Test datasets for dev/eval with energy and ground truth VAD

 - run_031_extract_xvectors_with_gt_diar.sh
    - Extracts x-vectors for spk detection test data using ground truth diarization
       - Test datasets for dev/eval with ground truth diarization

 - run_032_extract_xvectors_with_auto_diar.sh
    - Extracts x-vectors for spk detection test data using automatic diarization
       - Test datasets for dev/eval with automatic diarization based on ground truth VAD
       - Test datasets for dev/eval with automatic diarization based on energy VAD
    
 - run_033_extract_xvectors_for_tracking_with_gt_diar.sh
     - Extracts x-vectors for spk tracking test data using ground truth diarization
       - Test datasets for dev/eval with ground truth diarization
    
 - run_034_extract_xvectors_for_tracking_with_auto_diar.sh
    - Extracts x-vectors for spk tracking test data using automatic diarization
       - Test datasets for dev/eval with automatic diarization based on ground truth VAD
       - Test datasets for dev/eval with automatic diarization based on energy VAD

 - run_035_extract_xvectors_for_tracking_with_slid_win.sh
    - Extract x-vectors for spk tracking test data using a sliding window of 1.5s adn 50% overlap
        - Test datasets for dev/eval based on energy VAD

 - run_036_extract_xvectors_for_tracking_with_slid_win_gtvad.sh
    - Extract x-vectors for spk tracking test data using a sliding window of 1.5s adn 50% overlap
        - Test datasets for dev/eval based on grountruth VAD

 - run_040_train_spkdet_be.sh
    - Trains PLDA back-end for speaker detection
    - With/Without PLDA adaptation
       - Adapted to each dataset: babytrain, ami, chime5
    - Trained models are left in, for example:
    ```bash
    exp/be/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2_babytrain
    ```


 - run_041_eval_spkdet_be_wo_diar.sh
    - Evals speaker detection back-end for the three datasets without any speaker diarization
    - Two VADs: ground truth and energy VAD
    - Three back-end versions:
        - PLDA without domain adaptation
	- PLDA with domain adaptation
	- PLDA with domain adaptation + adaptive S-Norm
    - Calibrates scores
    - Score files and results (EER, DCF) are left in, for example:
    ```bash
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_cal_v1
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_gtvad_cal_v1
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_adapt_cal_v1
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_adapt_gtvad_cal_v1
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_adapt_snorm_cal_v1
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_adapt_snorm_gtvad_cal_v1
    ```
    - Results files for different enroll and test durations are named as: 
    ```bash
    jsalt19_spkdet_ami_eval_enr30_results
    jsalt19_spkdet_ami_eval_enr30_test15_results
    jsalt19_spkdet_ami_eval_enr30_test30_results
    jsalt19_spkdet_ami_eval_enr30_test5_results
    ```
    
	
 - run_042_eval_spkdet_be_with_gt_diar.sh
    - Evals speaker detection back-end for the three datasets using the ground truth diarization
    - Three back-end versions:
        - PLDA without domain adaptation
	- PLDA with domain adaptation
	- PLDA with domain adaptation + adaptive S-Norm
    - Calibrates scores
    - Score files and results (EER, DCF) are left in, for example:
    ```bash
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_adapt_gtdiar_cal_v1
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_adapt_snorm_gtdiar_cal_v1
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_gtdiar_cal_v1
    ```

 - run_043_eval_spkdet_be_with_auto_diar.sh
    - Evals speaker detection back-end for the three datasets using automatic diarization
    - Two diarization versions:
          - Ground truth
	  - Energy VAD
    - Three back-end versions:
        - PLDA without domain adaptation
	- PLDA with domain adaptation
	- PLDA with domain adaptation + adaptive S-Norm
    - Calibrates scores
        - Score files and results (EER, DCF) are left in, for example:
    ```bash
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_spkdetdiar_nnet2a.1.voxceleb_div2_thrbest_cal_v1
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_spkdetdiar_nnet2a.1.voxceleb_div2_thrbest_gtvad_cal_v1
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_adapt_spkdetdiar_nnet2a.1.voxceleb_div2_thrbest_cal_v1
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_adapt_spkdetdiar_nnet2a.1.voxceleb_div2_thrbest_gtvad_cal_v1
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_adapt_snorm_spkdetdiar_nnet2a.1.voxceleb_div2_thrbest_cal_v1
    exp/scores/2a.1.voxceleb_div2/lda200_splday150_v1_voxceleb_div2/plda_adapt_snorm_spkdetdiar_nnet2a.1.voxceleb_div2_thrbest_gtvad_cal_v1
    ```
 - run_04{4,5}_eval_spkdet_be_with_slid_wi*.sh
    - Evals speaker detection back-end for the three datasets using energy VAD and GTVAD

 - run_06{0,1,2}\_make_res_tables_spkdet\_\*.sh
    - Prints result tables for speaker detection for BabyTrain, AMI and SRI dataset.
    - Creates 5 tables for each dataset:
        - Automatic, VAD, automatic diarization, sliding window and sliding window with ground truth VAD.
	- Ground truth VAD and automatic diarization.
        - Ground truth VAD + diarization.
    - Easy to paste in Google spreadsheets:
        - Paste special -> values only.
	- Push split text to columns.


 - run_06{3,4,5,6}\_make_res_tables_spkdiar\_\*.sh
    - Prints result tables for speaker diarization for BabyTrain, AMI, CHiME5 and SRI dataset.
    - Creates 2 tables for each dataset:
        - Automatic VAD.
	- Ground truth VAD.
    - Easy to paste in Google spreadsheets:
        - Paste special -> values only.
	- Push split text to columns.
