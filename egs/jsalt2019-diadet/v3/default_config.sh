#Default configuration parameters for the experiment

#xvector training 
nnet_data=voxceleb_div2
nnet_vers=2a.1
nnet_name=2a.1.voxceleb_div2
nnet_num_epochs=3
nnet_dir=exp/xvector_nnet_$nnet_name


#diarization back-end
lda_diar_dim=120
plda_diar_data=voxceleb
be_diar_name=lda${lda_diar_dim}_plda_${plda_diar_data}
be_diar_babytrain_name=lda${lda_diar_dim}_plda_${plda_diar_data}_babytrain
be_diar_chime5_name=lda${lda_diar_dim}_plda_${plda_diar_data}_chime5
be_diar_ami_name=lda${lda_diar_dim}_plda_${plda_diar_data}_ami


#spkdet diarization vars
diar_thr=best
min_dur_spkdet_subsegs=4 # minimum duration for the diarization clusters used for spk detection
min_dur_track_subsegs=0.25 # minimum duration for the diarization clusters used for spk tracking
# automatic diarization rttms used for spkdet and tracking
rttm_babytrain_dir=./exp/diarization/$nnet_name/$be_diar_babytrain_name
rttm_chime5_dir=./exp/diarization/$nnet_name/$be_diar_chime5_name
rttm_ami_dir=./exp/diarization/$nnet_name/$be_diar_ami_name
rttm_sri_dir=$rttm_chime5_dir
spkdet_diar_name=spkdetdiar_nnet${nnet_name}_thr${diar_thr}
track_diar_name=trackdiar_nnet${nnet_name}_thr${diar_thr}


#spkdet back-end
lda_dim=200
plda_y_dim=150
plda_z_dim=200

plda_data=voxceleb_div2
plda_type=splda
plda_label=${plda_type}y${plda_y_dim}_v1

be_name=lda${lda_dim}_${plda_label}_${plda_data}
be_babytrain_name=${be_name}_babytrain
be_chime5_name=${be_name}_chime5
be_ami_name=${be_name}_ami

#adaptation weights for plda between-class and within-class covs
w_B_babytrain=0
w_W_babytrain=0
w_B_chime5=0
w_W_chime5=0.1
w_B_ami=0.1
w_W_ami=0.25

#score norm, number of cohort recordings
#ncoh_babytrain=300
#ncoh_chime5=300
#ncoh_ami=300
ncoh=500
