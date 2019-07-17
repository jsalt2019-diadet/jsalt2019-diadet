#!/usr/bin/env python
"""
 Copyright 2018 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import argparse
import time
import logging

import numpy as np
from six.moves import xrange

import torch
import torch.nn as nn

from hyperion.hyp_defs import config_logger
from hyperion.io import SequentialAudioReader as AR
from hyperion.io import SequentialDataReaderFactory as DRF
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import compression_methods
from hyperion.feats import MFCC

#define dummy class
class DummyNet(nn.Module):

    def __init__(self, dim):
        super(DummyNet, self).__init__()
        self.weight = nn.Parameter(torch.eye(dim))

    def forward(self, x):
        return x.matmul(self.weight)


def apply_nnet(x, model, chunk_size, context, device):
    x = x.astype('float32')

    if chunk_size == 0:
        x = torch.tensor(x).to(device)
        return model(x).detach().numpy()

    half_context = int((context - 1)/2)
    chunk_shift = chunk_size - (context-1)
    chunk_size_out = chunk_size - half_context
    
    y = np.zeros_like(x, dtype='float32')
    num_chunks = int(np.ceil(x.shape[0]/chunk_shift))
    tbeg_in = 0
    tbeg_out = 0
    for i in range(num_chunks):
        tend_in = min(tbeg_in + chunk_size, x.shape[0])

        x_i = torch.tensor(x[tbeg_in:tend_in]).to(device)
        y_i = model(x_i).detach().numpy()
        if i == 0:
            tend_out = min(tbeg_out + chunk_size, x.shape[0])
            y[tbeg_out:tend_out] = y_i
            tbeg_out =+ chunk_size_out
        else:
            tend_out = min(tbeg_out + chunk_size_out, x.shape[0])
            y[tbeg_out:tend_out] = y_i[half_context:]
            tbeg_out += chunk_shift

        tbeg_in += chunk_shift

    return y


def compute_mfcc_feats(input_path, output_path,
                       compress, compression_method, write_num_frames, 
                       use_gpu, nn_model_path, chunk_size, context,
                       **kwargs):

    #open device
    if  use_gpu and torch.cuda.is_available():
        logging.info('CUDA_VISIBLE_DEVICES=%s' % os.environ['CUDA_VISIBLE_DEVICES'])
        logging.info('init gpu device')
        device = torch.device('cuda', 0)
        torch.tensor([0]).to(device)
    else:
        logging.info('init cpu device')
        device = torch.device('cpu')

    mfcc_args = MFCC.filter_args(**kwargs)
    mfcc = MFCC(**mfcc_args)
    # PUT YOUR NNET MODEL HERE!!!!
    enhancer = DummyNet(mfcc.num_filters)
    # we should load the weights here
    # enhancer.load_state_dict(torch.load(nn_model_path))
    enhancer.to(device)
    enhancer.eval()

    if mfcc.input_step == 'wave':
        input_args = AR.filter_args(**kwargs)
        reader = AR(input_path, **input_args)
    else:
        input_args = DRF.filter_args(**kwargs)
        reader = DRF.create(input_path, **input_args)

    writer = DWF.create(output_path, scp_sep=' ',
                        compress=compress,
                        compression_method=compression_method)

    if write_num_frames is not None:
        f_num_frames = open(write_num_frames, 'w')
    
    for data in reader:
        if mfcc.input_step == 'wave':
            key, x, fs = data
        else:
            key, x = data
        logging.info('Extracting filter-banks for %s' % (key))
        t1 = time.time()
        y = mfcc.compute(x)

        #we apply dummy identity network to fb
        logging.info('Running enhancement network')
        y = apply_nnet(y, enhancer, chunk_size, context, device)

        dt = (time.time() - t1)*1000
        rtf = mfcc.frame_shift*y.shape[0]/dt
        logging.info('Extracted filter-banks for %s num-frames=%d elapsed-time=%.2f ms. real-time-factor=%.2f' %
                     (key, y.shape[0], dt, rtf))
        writer.write([key], [y])
        
        if write_num_frames is not None:
            f_num_frames.write('%s %d\n' % (key, y.shape[0]))

        mfcc.reset()
            
    if write_num_frames is not None:
        f_num_frames.close()
    

if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Compute filter-bank features and enhance with pytorch model')

    parser.add_argument('--input', dest='input_path', required=True)
    parser.add_argument('--output', dest='output_path', required=True)
    parser.add_argument('--write-num-frames', dest='write_num_frames', default=None)

    DRF.add_argparse_args(parser)
    MFCC.add_argparse_args(parser)
    parser.add_argument('--compress', dest='compress', default=False, action='store_true', help='Compress the features')
    parser.add_argument('--compression-method', dest='compression_method', default='auto',
                        choices=compression_methods, help='Compression method')
    parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int,
                        help='Verbose level')
    parser.add_argument('--use-gpu', dest='use_gpu', default=False, 
                        action='store_true', help='uses gpu to apply pytorch model')
    parser.add_argument('--nn-model-path', dest='nn_model_path', required=True)
    parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=0)
    parser.add_argument('--context', dest='context', type=int, default=0)

    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    args.output_step='logfb'
    logging.debug(args)

    
    compute_mfcc_feats(**vars(args))
    
