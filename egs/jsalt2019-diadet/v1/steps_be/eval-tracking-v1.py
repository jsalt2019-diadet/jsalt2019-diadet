#!/usr/bin/env python
"""
  Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  

  Evals PLDA LLR
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
import pandas as pd

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.utils import SCPList, TrialNdx, TrialScores, ExtSegmentList, RTTM
from hyperion.helpers.tracking_data_reader import TrackingDataReader as TDR
from hyperion.helpers import PLDAFactory as F
from hyperion.transforms import TransformList


def flatten_segment_scores(ndx_seg, scores):

    scores = scores.align_with_ndx(ndx_seg)
    idx=(ndx_seg.trial_mask.T == True).nonzero()
    new_segment_ids = []
    segment_ids = []
    model_ids = []
    flat_scores = np.zeros((len(idx[0]),), dtype=float)
    k = 0
    for item in zip(idx[0], idx[1]):
        model_ids.append(ndx_seg.model_set[item[1]])
        segment_ids.append(ndx_seg.seg_set[item[0]])
        new_segment_ids.append('%s-%08d' % (ndx_seg.seg_set[item[0]],k))
        flat_scores[k] = scores.scores[item[1], item[0]]
        k +=1

    new_segment_ids = np.array(new_segment_ids)
    segment_ids = np.array(segment_ids)
    model_ids = np.array(model_ids)

    return new_segment_ids, segment_ids, model_ids, flat_scores


def prepare_output_ext_segments(ext_segments_in, new_ext_segment_ids, ext_segment_ids, model_ids, scores):

    df_map = pd.DataFrame({'new_ext_segment_id': new_ext_segment_ids, 'ext_segment_id': ext_segment_ids})
    new_segments = pd.merge(ext_segments_in.segments, df_map)
    new_segments.drop(columns=['ext_segment_id'], inplace=True)
    new_segments.rename(index=str, columns={'new_ext_segment_id': 'ext_segment_id'}, inplace=True)
    new_segments.sort_values(by=['file_id','tbeg'], inplace=True)
    ext_segments = ExtSegmentList(new_segments)
    ext_segments.assign_names(new_ext_segment_ids, model_ids, scores)
    print(model_ids)
    print(ext_segments.segments)
    print(ext_segments.ext_segments)
    return ext_segments




def tracking_plda(iv_file, ndx_file, enroll_file, segments_file,
              preproc_file,
              model_file, rttm_file, plda_type,
              **kwargs):
    
    logging.info('loading data')
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    tdr = TDR(iv_file, ndx_file, enroll_file, segments_file, preproc)
    x_e, x_t, enroll, ndx_seg, ext_segments = tdr.read()

    logging.info('loading plda model: %s' % (model_file))
    model = F.load_plda(plda_type, model_file)
    
    t1 = time.time()
    
    logging.info('computing llr')
    scores = model.llr_1vs1(x_e, x_t)
    
    dt = time.time() - t1
    num_trials = len(enroll) * x_t.shape[0]
    logging.info('scoring elapsed time: %.2f s. elapsed time per trial: %.2f ms.'
          % (dt, dt/num_trials*1000))

    scores = TrialScores(enroll, ndx_seg.seg_set, scores)
    new_ext_segment_ids, ext_segment_ids, model_ids, scores = flatten_segment_scores(ndx_seg, scores)
    new_ext_segments = prepare_output_ext_segments(
        ext_segments, new_ext_segment_ids, ext_segment_ids, model_ids, scores)
    new_ext_segments.save(rttm_file + '_es')
    rttm = RTTM.create_spkdiar_from_ext_segments(new_ext_segments)
    rttm.save(rttm_file)

    
    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Eval PLDA for tracking spks')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--ndx-file', dest='ndx_file', required=True)
    parser.add_argument('--enroll-file', dest='enroll_file', required=True)
    parser.add_argument('--segments-file', dest='segments_file', required=True)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)

    TDR.add_argparse_args(parser)
    F.add_argparse_eval_args(parser)

    parser.add_argument('--rttm-file', dest='rttm_file', required=True)
    parser.add_argument('-v', '--verbose', dest='verbose', default=1,
                        choices=[0, 1, 2, 3], type=int)
    
    args=parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    tracking_plda(**vars(args))

            
