#!/usr/bin/python

import sys, re
import numpy as np
import kaldi_io

def get_zca_whitening_matrix(vectors):
    #EPS = 10e-0 # variance flooring
    EPS = 0.

    Cov  = np.cov(vectors)   # covariance
    Cov  = Cov + np.eye(Cov.shape[0])

    d, E = np.linalg.eigh(Cov) # eigen-value decomosition
    D    = np.diag(1. / np.sqrt(d + EPS)) # D = d^{-1/2}
    W    = np.dot(np.dot(E, D), E.T) # W = EDE^t = Ed^-{1/2}E^t
    return W

def zca_whitening(vectors, W):
    return np.dot(W, vectors)

def zca_recoloring(vectors, W):
    return np.dot(np.linalg.inv(W), vectors)

GET_MATRIX = 1
CONVERT    = 2
RECOLOR    = 3

if len(sys.argv) == 3:
    mode = GET_MATRIX
    input_vec_file = sys.argv[1]
    matrix_file    = sys.argv[2]

    print >> sys.stderr, "mode: output ZCA whitening matrix"
    print >> sys.stderr, "  input_vec_file  ", input_vec_file
    print >> sys.stderr, "  matrix_file     ", matrix_file
    #print >> sys.stderr, ""

elif len(sys.argv) == 4:
    mode = CONVERT
    input_vec_file  = sys.argv[1]
    matrix_file     = sys.argv[2]
    output_vec_file = sys.argv[3]

    print >> sys.stderr, "mode: convert vectors with ZCA whitening matrix"
    print >> sys.stderr, "  input_vec_file   ", input_vec_file
    print >> sys.stderr, "  matrix_file      ", matrix_file
    print >> sys.stderr, "  output_vec_file  ", output_vec_file
    #print >> sys.stderr, ""

elif len(sys.argv) == 5:
    mode = RECOLOR
    input_vec_file  = sys.argv[1]
    wh_matrix_file  = sys.argv[2]
    co_matrix_file  = sys.argv[3]
    output_vec_file = sys.argv[4]

    print >> sys.stderr, "mode: convert vectors with ZCA whitening matrix & recoloring matrix"
    print >> sys.stderr, "  input_vec_file   ", input_vec_file
    print >> sys.stderr, "  wh_matrix_file   ", wh_matrix_file
    print >> sys.stderr, "  co_matrix_file   ", co_matrix_file
    print >> sys.stderr, "  output_vec_file  ", output_vec_file
    #print >> sys.stderr, ""

else:
    print >> sys.stderr, "Usage:$0 ark:input.ark zca_whitening_matrix"
    print >> sys.stderr, "         scp:input.scp zca_whitening_matrix"
    print >> sys.stderr, "         ark:input.ark zca_whitening_matrix ark:output.ark"
    sys.exit()

# load input
if re.search('^ark:', input_vec_file):
    if input_vec_file[-1] == "-":
        vectors = list(kaldi_io.read_vec_flt_ark(sys.stdin))
    else:
        vectors = list(kaldi_io.read_vec_flt_ark(input_vec_file))
elif re.search('^scp:', input_vec_file):
    if input_vec_file[-1] == "-":
        vectors = kaldi_io.read_vec_flt_scp(sys.stdin)
    else:
        vectors = kaldi_io.read_vec_flt_scp(input_vec_file)
else:
    print >> sys.stderr, '[ERROR]: input_vec_file should be "ark:vector.ark" or "scp:vector.scp"'
    sys.exit()

utt_ids =          [uid for uid, vec in vectors]
vectors = np.array([vec for uid, vec in vectors], dtype=np.float32).T

# input i-vectors must be centerized !!!
# print >> sys.stderr, "NOTE: input i-vectors must be centerized !!!"

if mode == GET_MATRIX:
    W = get_zca_whitening_matrix(vectors)
    kaldi_io.write_mat(matrix_file, W)

elif mode == CONVERT:
    W = kaldi_io.read_mat(matrix_file)

    vectors = zca_whitening(vectors, W).T
    vectors = np.array(vectors, dtype=np.float32)
elif mode == RECOLOR:
    W = kaldi_io.read_mat(wh_matrix_file)
    V = kaldi_io.read_mat(co_matrix_file)

    vectors = zca_whitening(vectors, W)
    vectors = zca_recoloring(vectors, V).T
    vectors = np.array(vectors, dtype=np.float32)
    

if mode == CONVERT or mode == RECOLOR:
    # save (binary ark...)
    if re.search('^ark:', output_vec_file):
        if output_vec_file[-1] == "-":
            fd = sys.stdout
        else:
            fd = kaldi_io.open_or_fd(output_vec_file, "wb")
        for i, uid in enumerate(utt_ids):
            vec = vectors[i]
            kaldi_io.write_vec_flt(fd, vec, uid)
        fd.close()
    else:
        print >> sys.stderr, '[ERROR]: output_vec_file should be "ark:vector.ark"'
        sys.exit()
