#Default configuration parameters for the experiment

#xvector training 
nnet_data=voxceleb_div2
nnet_vers=2a.1
nnet_name=2a.1.voxceleb_div2
nnet_num_epochs=3
nnet_dir=exp/xvector_nnet_$nnet_name


#spkdet back-end
lda_dim=200
plda_y_dim=150
plda_z_dim=200

plda_data=voxceleb_div2
plda_type=splda
plda_label=${plda_type}y${plda_y_dim}_v1

be_name=lda${lda_dim}_${plda_label}_${plda_data}


#enhancement
# context aggregation network with residual x+logsigmoid(can(x)), linear context
enh_name=cycgan_1
py_mfcc_enh=steps_pyfe/pytorch-compute-mfcc-feats-cycgan-v3.py
#dummy enhancement network parameters
enh_nnet=exp/se_models/cycgan_1/checkpoint.epoch50.tgt2src_gen.pth
enh_context=55
enh_chunk_size=127

# data to train plda for enhancement
enh_train=false
plda_data_enh=${plda_data}
be_enh_name=lda${lda_dim}_${plda_label}_${plda_data_enh}


