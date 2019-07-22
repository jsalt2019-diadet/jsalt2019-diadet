"""Various utility functions."""
from __future__ import print_function
from __future__ import unicode_literals
import numbers
import os
import sndhdr
import struct
import sys

import numpy as np
import scipy.signal


EPS = 1e-8


def warn(msg):
    """Print warning message to STERR."""
    msg = 'WARN: %s' % msg
    print(msg, file=sys.stderr)


def error(msg):
    """Print warning message to STERR."""
    msg = 'ERROR: %s' % msg
    print(msg, file=sys.stderr)


def stft(x, window, n_per_seg=512, noverlap=256):
    """Return short-time Fourier transform (STFT) for signal.

    Parameters
    ----------
    x : ndarray, (n_samps,)
        Input signal.

    window : ndarray, (wl,)
        Array of weights to use when windowing the signal.

    n_per_seg : int, optional
    """
    if len(window) != n_per_seg:
        raise ValueError('window length must equal n_per_seg')
    x = np.array(x)
    nadd = noverlap - (len(x) - n_per_seg) % noverlap
    x = np.concatenate((x, np.zeros(nadd)))
    hop_size = n_per_seg - noverlap
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_size, n_per_seg)
    strides = x.strides[:-1] + (hop_size * x.strides[-1], x.strides[-1])
    x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    x = x * window
    result = np.fft.rfft(x, n=n_per_seg)
    return result


def wav2logspec(x, window, n_per_seg=512, noverlap=256):
    """TODO"""
    y = stft(x, window, n_per_seg=n_per_seg, noverlap=noverlap)
    return np.log(np.square(abs(y)) + EPS)


def MY_logspec2wav(lps, wave, window, n_per_seg=512, noverlap=256):
    "Convert log-power spectrum back to time domain."""
    z = stft(wave, window)
    angle = z / (np.abs(z) + EPS) # Recover phase information
    x = np.sqrt(np.exp(lps)) * angle
    x = np.fft.irfft(x)
    y = np.zeros((len(x) - 1) * noverlap + n_per_seg)
    C1 = window[0:256]
    C2 = window[0:256] + window[256:512]
    C3 = window[256:512]
    y[0:noverlap] = x[0][0:noverlap] / C1
    for i in range(1, len(x)):
        y[i*noverlap:(i + 1)*noverlap] = (x[i-1][noverlap:n_per_seg] + x[i][0:noverlap]) / C2
    y[-noverlap:] = x[len(x)-1][noverlap:] / C3
    return np.int16(y[0:len(wave)])


def logspec2wav(lps, ref_wav, window, n_per_seg=512, noverlap=256):
    "Convert log-power spectrum back to time domain."""
    hop_size = n_per_seg - noverlap
    assert len(window) % hop_size == 0, "The constraint of “Constant OverLap Add” (COLA) is not satisfied!"
    ref_stft = stft (ref_wav, window, n_per_seg=n_per_seg, noverlap=noverlap) 
    angle=ref_stft/ (np.abs(ref_stft) + EPS ) # Recover phase information
    mag_x=np.sqrt(np.exp(lps))* angle
    frames=np.fft.irfft (mag_x)   
    back_wav=np.zeros((len(frames) -1) * noverlap + n_per_seg)
    C1= window[0: hop_size]
    C2= window[hop_size:]  + window[:noverlap]
    C3= window[noverlap]
    back_wav[0:hop_size] = frames[0][0:hop_size] / C1 #

    for i in range (1, len(frames)):
        back_wav[i*hop_size : i*hop_size+noverlap] = (frames[i-1][hop_size:] + frames[i][:noverlap])/C2
        back_wav[i*hop_size+noverlap : i*hop_size+ n_per_seg] = frames[i][noverlap:] /C3 
    back_wav[-hop_size:] =frames[len(frames) -1][noverlap:] /C3  
    return np.int16(back_wav[0: len(ref_wav)])  

def expand_frames( data, context_window):
    if not len(context_window):
        raise Exception("context_window needs both left and right infos!")
    left_context = context_window[0]
    right_context = context_window[1]
    window_len = sum(context_window) + 1
    data_left = np.tile( data[0,:],  (left_context,1) )
    data_right = np.tile( data[-1,:], (right_context,1) )
    data_full = np.concatenate( ( data_left, data, data_right),axis=0)

    frames = data.shape[0]
    dim = data.shape[1]
    expand_data = np.zeros( ( frames, dim* window_len ))
    for i in  range(0,frames):
        expand_data[i,:] = data_full[ i:i+window_len].reshape(-1)
    return expand_data


MAX_PCM_VAL = 32767
def peak_normalization(x):
    """Perform peak normalization."""
    norm = x.astype(float)
    norm = norm / max(abs(norm)) * MAX_PCM_VAL
    return norm.astype(int)


def read_htk(filename):
    """Return features from HTK file a 2-D numpy array."""
    with open(filename, 'rb') as f:
        # Read header
        n_samples, samp_period, samp_size, parm_kind = struct.unpack(
            '>iihh', f.read(12))

        # Read data
        data = struct.unpack(
            '>%df' % (n_samples * samp_size / 4), f.read(n_samples * samp_size))

        return n_samples, samp_period, samp_size, parm_kind, data


def write_htk(filename, feature, samp_period, parm_kind):
    """Write array of frame-level features to HTK binary file."""
    with open(filename, 'wb') as f:
        # Write header
        n_samples = feature.shape[0]
        samp_size = feature.shape[1] * 4
        f.write(struct.pack('>iihh', n_samples, samp_period, samp_size, parm_kind))
        f.write(struct.pack('>%df' % (n_samples * samp_size / 4), *feature.ravel()))


VALID_VAD_SRS = {8000, 16000, 32000, 48000}
VALID_VAD_FRAME_LENGTHS = {10, 20, 30}
VALID_VAD_MODES = {0, 1, 2, 3}



def get_segments(vad_info, fs):
    """Convert array of VAD labels into segmentation."""
    vad_index = np.where(vad_info == 1.0) # Find the speech index.
    vad_diff = np.diff(vad_index)

    vad_temp = np.zeros_like(vad_diff)
    vad_temp[np.where(vad_diff == 1)] = 1
    vad_temp = np.column_stack((np.array([0]), vad_temp, np.array([0])))
    final_index = np.diff(vad_temp)

    starts = np.where(final_index == 1)
    ends = np.where(final_index == -1)

    sad_info = np.column_stack([starts[1], ends[1]])
    vad_index = vad_index[0]

    segments = np.zeros_like(sad_info, dtype=np.float)
    for i in range(sad_info.shape[0]):
        segments[i][0] = float(vad_index[sad_info[i][0]]) / fs
        segments[i][1] = float(vad_index[sad_info[i][1]] + 1) / fs

    return segments  # Present in seconds.


def write_segments(fn, segs, n_digits=3, label=''):
    """Write segmentation to file."""
    fmt_str = '%%.%df %%.%df %%s\n' % (n_digits, n_digits)
    with open(fn, 'wb') as f:
        for onset, offset in segs:
            line = fmt_str % (onset, offset, label)
            f.write(line.encode('utf-8'))


def listdir(dirpath, abspath=True, ext=None):
    """List contents of directory."""
    fns = os.listdir(dirpath)
    if ext is not None:
        fns = [fn for fn in fns if fn.endswith(ext)]
    if abspath:
        fns = [os.path.abspath(os.path.join(dirpath, fn))
               for fn in fns]
    fns = sorted(fns)
    return fns


def load_script_file(fn, ext=None):
    """Load HTK script file of paths."""
    with open(fn, 'rb') as f:
        paths = [line.decode('utf-8').strip() for line in f]
    paths = sorted(paths)
    if ext is not None:
        filt_paths = []
        for path in paths:
            if not path.endswith(ext):
                warn('Skipping file "%s" that does not match extension "%s"' %
                     (path, ext))
                continue
            filt_paths.append(path)
        paths = filt_paths
    return paths


def xor(x, y):
    """Return truth value of ``x`` XOR ``y``."""
    return bool(x) != bool(y)


def is_wav(fn):
    """Returns True if ``fn`` is a WAV file."""
    hinfo = sndhdr.what(fn)
    if hinfo is None:
        return False
    elif hinfo[0] != 'wav':
        return False
    return True


def get_sr(fn):
    """Return sample rate in Hz of WAV file."""
    if not is_wav(fn):
        raise ValueError('File "%s" is not a valid WAV file.' % fn)
    hinfo = sndhdr.what(fn)
    return hinfo[1]


def get_num_channels(fn):
    """Return number of channels present in  WAV file."""
    if not is_wav(fn):
        raise ValueError('File "%s" is not a valid WAV file.' % fn)
    hinfo = sndhdr.what(fn)
    return hinfo[2]


def get_bitdepth(fn):
    """Return bitdepth of WAV file."""
    if not is_wav(fn):
        raise ValueError('File "%s" is not a valid WAV file.' % fn)
    hinfo = sndhdr.what(fn)
    return hinfo[4]
