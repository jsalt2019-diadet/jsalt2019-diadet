#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

  Trains Backend for jsalt19 spkdet
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import os
import argparse
import time
import logging

import numpy as np

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.helpers import VectorClassReader as VCR
from hyperion.helpers import VectorReader as VR
from hyperion.transforms import TransformList, LDA, LNorm
from hyperion.helpers import PLDAFactory as F


def train_be(iv_file, train_list,
             adapt_iv_file, adapt_list,
             lda_dim, 
             plda_type, y_dim, z_dim,
             epochs, ml_md, md_epochs,
             w_mu, w_B, w_W,
             output_path, **kwargs):

    # Read data
    logging.info('loading data')
    vcr_args = VCR.filter_args(**kwargs)
    vcr_train = VCR(iv_file, train_list, None, **vcr_args)
    x, class_ids = vcr_train.read()

    # Train LDA
    logging.info('train LDA')
    t1 = time.time()
    lda = LDA(lda_dim=lda_dim, name='lda')
    lda.fit(x, class_ids)

    x_lda = lda.predict(x)
    logging.info('LDA elapsed time: %.2f s.' % (time.time()-t1))

    # Train centering and whitening
    logging.info('train length norm')
    t1 = time.time()
    lnorm = LNorm(name='lnorm')    
    lnorm.fit(x_lda)

    x_ln = lnorm.predict(x_lda)
    logging.info('length norm elapsed time: %.2f s.' % (time.time()-t1))
    
    # Train PLDA
    logging.info('train PLDA')
    t1 = time.time()
    plda = F.create_plda(plda_type, y_dim=y_dim, z_dim=z_dim,
                         name='plda')
    elbo = plda.fit(x_ln, class_ids, 
                    epochs=epochs, ml_md=ml_md, md_epochs=md_epochs)
    logging.info('PLDA elapsed time: %.2f s.' % (time.time()-t1))

    # Save models
    logging.info('saving models')
    preproc = TransformList(lda)
    preproc.append(lnorm)

    if not os.path.exists(output_path):
        os.makedirs(ouput_path)

    preproc.save(output_path + '/lda_lnorm.h5')
    plda.save(output_path + '/plda.h5')

    num = np.arange(epochs)
    elbo = np.vstack((num, elbo)).T
    np.savetxt(output_path + '/elbo.csv', elbo, delimiter=',')
 
    #adaptation
    vcr = VCR(adapt_iv_file, adapt_list, None)
    x, class_ids = vcr.read()
    x_lda = lda.predict(x)
    lnorm.update_T = False
    lnorm.fit(x_lda)

    preproc = TransformList(lda)
    preproc.append(lnorm)

    preproc.save(output_path + '/lda_lnorm_adapt.h5')
    
    x_ln = lnorm.predict(x_lda)

    plda_adapt = plda.copy()
    
    elbo = plda.fit(x_ln, class_ids, epochs=epochs)
    plda_adapt.weighted_avg_model(plda, w_mu, w_B, w_W)
    plda_adapt.save(output_path + '/plda_adapt.h5')

    num = np.arange(epochs)
    elbo = np.vstack((num, elbo)).T
    np.savetxt(output_path + '/elbo_adapt.csv', elbo, delimiter=',')


    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Train Back-end with adaptation')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--adapt-iv-file', dest='adapt_iv_file', required=True)
    parser.add_argument('--adapt-list', dest='adapt_list', required=True)

    VCR.add_argparse_args(parser)
    F.add_argparse_train_args(parser)
    
    parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument('--lda-dim', dest='lda_dim', type=int,
                        default=None)
    parser.add_argument('--w-mu', dest='w_mu', type=float,
                        default=1)
    parser.add_argument('--w-b', dest='w_B', type=float,
                        default=1)
    parser.add_argument('--w-w', dest='w_W', type=float,
                        default=1)

    parser.add_argument('-v', '--verbose', dest='verbose', default=1,
                        choices=[0, 1, 2, 3], type=int)

    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)
    
    train_be(**vars(args))

            
