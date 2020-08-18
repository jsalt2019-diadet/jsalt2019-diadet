#!/usr/bin/env python3

# Copyright 2013-2017 Lukas Burget (burget@fit.vutbr.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Revision History
#   L. Burget   16/07/13 01:00AM - original version

import numpy as np
import VB_cldiarization
import argparse

parser = argparse.ArgumentParser(description='VB clustering Wrapper')
parser.add_argument('xvec_dir', type=str, help='Subset data directory')
parser.add_argument('init_labels_filename', type=str, 
                    help='The labels file to initialize the VB system, usually the AHC cluster result')
parser.add_argument('output_dir', type=str, help='Output directory')
parser.add_argument('plda_mean', type=str, help='Path of the plda mean')
parser.add_argument('plda_psi', type=str, help='Path of the plda psi')

parser.add_argument('--max-speakers', type=int, default=10,
                    help='Maximum number of speakers expected in the utterance (default: 10)')
parser.add_argument('--max-iters', type=int, default=10,
                    help='Maximum number of algorithm iterations (default: 10)')
parser.add_argument('--downsample', type=int, default=25,
                    help='Perform diarization on input downsampled by this factor (default: 25)')
parser.add_argument('--alphaQInit', type=float, default=100.0,
                    help='Dirichlet concentraion parameter for initializing q')
parser.add_argument('--sparsityThr', type=float, default=0.001,
                    help='Set occupations smaller that this threshold to 0.0 (saves memory as \
                    the posteriors are represented by sparse matrix)')
parser.add_argument('--epsilon', type=float, default=1e-6,
                    help='Stop iterating, if obj. fun. improvement is less than epsilon')
parser.add_argument('--minDur', type=int, default=1,
                    help='Minimum number of frames between speaker turns imposed by linear \
                    chains of HMM states corresponding to each speaker. All the states \
                    in a chain share the same output distribution')
parser.add_argument('--loopProb', type=float, default=0.9,
                    help='Probability of not switching speakers between frames')
parser.add_argument('--statScale', type=float, default=0.2,
                    help='Scale sufficient statiscits collected using UBM')
parser.add_argument('--llScale', type=float, default=1.0,
                    help='Scale UBM likelihood (i.e. llScale < 1.0 make atribution of \
                    frames to UBM componets more uncertain)')
parser.add_argument('--channel', type=int, default=0,
                    help='Channel information in the rttm file')
parser.add_argument('--initialize', type=int, default=1,
                    help='Whether to initalize the speaker posterior')

args = parser.parse_args()
print(args)

# Load the plda mean and psi
mean = np.loadtxt(args.plda_mean)
psi = np.loadtxt(args.plda_psi)
    
m = mean.reshape(*(1,120))
iE = (np.ones(120)).reshape(*(1,120)) #indetity matrix instead of within matrix 
w = 1 # 1 GMM componet
V = np.diag(np.sqrt(psi)).reshape(-1,*m.shape) #square root of PLDA psi matrix   
    
# Load the xvector
X=np.loadtxt(args.xvec_dir)
    
# Load the initialization labels
ref=np.loadtxt(args.init_labels_filename, dtype=int)

VtiEV = VB_diarization.precalculate_VtiEV(V, iE)
       
# Initialize the posterior of each speaker based on the clustering result.
if args.initialize:
    q = VB_diarization.frame_labels2posterior_mx(ref, args.max_speakers)
else:
    q = None
        
q_out, sp_out, L_out = VB_diarization.VB_diarization(X, m, iE, w, V, sp=None, q=q, maxSpeakers=args.max_speakers, maxIters=args.max_iters, VtiEV=None, downsample=args.downsample, alphaQInit=args.alphaQInit, sparsityThr=args.sparsityThr, epsilon=args.epsilon, minDur=args.minDur, loopProb=args.loopProb, statScale=args.statScale, llScale=args.llScale, ref=None, plot=False)
predicted_label = np.argmax(q_out, 1) + 1

np.savetxt(args.output_dir,predicted_label,fmt="%d")

    
