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

diar_thr=-0.9
min_dur=10
rttm_dir=./exp/diarization/$nnet_name/$be_diar_name
diar_name=diar${nnet_name}_thr${diar_thr}
