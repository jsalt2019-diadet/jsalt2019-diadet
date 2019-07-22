#!/usr/bin/env python
"""Perform speech enhancement for audio stored in WAV files.

This script performs speech enhancement of audio using a deep-learning based
enhancement model (Lei et al, 2018; Gao et al, 2018; Lei et al, 2017). To perform
enhancement for all WAV files under the directory ``wav_dir/`` and write the
enhanced audio to ``se_wav_dir/`` as WAV files:

    python main_denoising.py --wav_dir wav_dir --output_dir se_wav_dir

For each file with the ``.wav`` extension under ``wav_dir/``, there will now be
a corresponding enhanced version under ``se_wav_dir``.

Alternately, you may specify the files to process via a script file of paths to
WAV files with one path per line:

    /path/to/file1.wav
    /path/to/file2.wav
    /path/to/file3.wav
    ...

The GPU using chunks that are 10 minutes in duration. This should use at
most 8 GB of GPU memory.

References
----------
- Sun, Lei, et al. (2018). "Speaker diarization with enhancing speech for the First DIHARD
 Challenge." Proceedings of INTERSPEECH 2018. 2793-2797.
- Gao, Tian, et al. (2018). "Densely connected progressive learning for LSTM-based speech
  enhancement." Proceedings of ICASSP 2018.
- Sun, Lei, et al. (2017). "Multiple-target deep learning for LSTM-RNN based speech enhancement."
  Proceedings of the Fifth Joint Workshop on Hands-free Speech Communication and Microphone
  Arrays.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import math
import os
import sys
import traceback
import pdb
import numpy as np
import scipy.io.wavfile as wav_io
import scipy.io as sio
from model import LSTM_SE_PL_Dense_MTL
import utils
import torch
HERE = os.path.abspath(os.path.dirname(__file__))

SR = 16000 # Expected sample rate (Hz) of input WAV.
NUM_CHANNELS = 1 # Expected number of channels of input WAV.
BITDEPTH = 16 # Expected bitdepth of input WAV.
WL = 512 # Analysis window length in samples for feature extraction.
WL2 = WL // 2
NFREQS = 257 # Number of positive frequencies in FFT output.
if torch.cuda.is_available():
    gpu_id = int(os.popen('free-gpu').read())
    torch.cuda.set_device(gpu_id)



def denoise_wav(src_wav_file, dest_wav_file, global_mean, global_var, use_gpu,
                truncate_minutes, mode, stage_select):
    """Apply speech enhancement to audio in WAV file.

    Parameters
    ----------
    src_wav_file : str
        Path to WAV to denosie.

    dest_wav_file : str
        Output path for denoised WAV.

    global_mean : ndarray, (n_feats,)
        Global mean for LPS features. Used for CMVN.

    global_var : ndarray, (n_feats,)
        Global variances for LPS features. Used for CMVN.

    use_gpu : bool, optional
        If True and GPU is available, perform all processing on GPU.
        (Default: True)

    truncate_minutes: float
        Maximimize size in minutes to process at a time. The enhancement will
        be done on chunks of audio no greather than ``truncate_minutes``
        minutes duration.
    """
    # Read noisy audio WAV file. As scipy.io.wavefile.read is FAR faster than
    # librosa.load, we use the former.
    rate, wav_data = wav_io.read(src_wav_file)

    if mode == 1:
        print("###Selecting the estimated ideal-ratio-masks in mode 1 (more conservative).###")
    elif mode == 2:
        print("###Selecting the estimated log-power-spec features in mode 2 (more agressive).###")
    elif mode == 3:
        print("###Selecting both estimated IRM and LPS outputs with equal weights in mode 3 (trade-off).###")
       
    # Apply peak-normalization.
    wav_data = utils.peak_normalization(wav_data)

    # Perform denoising in chunks of size chunk_length samples.
    chunk_length = int(truncate_minutes*rate*60)
    total_chunks = int(math.ceil(wav_data.size / chunk_length))
    data_se = [] # Will hold enhanced audio data for each chunk.   
    
    model_pth = os.path.join(HERE, '1000h_se.pth')
    if not os.path.exists(model_pth):
        cmd = "cp  {} {} ".format('/export/fs01/jsalt19/leisun/speech_enhancement/speech_denoising_pytorch/model/1000h_se.pth', model_pth )
        os.system(cmd)
    
    nnet = LSTM_SE_PL_Dense_MTL(257, 7, 1024, 3, 257,'false')
    nnet.load_state_dict(torch.load(model_pth))
    nnet = nnet.cuda()
    nnet.eval()

    for i in range(1, total_chunks + 1):
        # Get samples for this chunk.
        bi = (i-1)*chunk_length # Index of first sample of this chunk.
        ei = bi + chunk_length # Index of last sample of this chunk + 1.
        temp = wav_data[bi:ei]
        print('Processing file: %s, segment: %d/%d.' %(src_wav_file, i, total_chunks))

        # Skip denoising if chunk is too short.
        if temp.shape[0] < WL2:
            data_se.append(temp)
            continue

        # Extract LPS features from waveform.
        noisy_htkdata = utils.wav2logspec(temp, window=np.hamming(WL))
        # frame expandation in the input
        noisy_htkdata_expand = utils.expand_frames(noisy_htkdata, [3,3] ) 
            
        input = torch.from_numpy (  (noisy_htkdata_expand - global_mean) / (global_var) ) 
        lps_outputs, irm_outputs= nnet( torch.unsqueeze( input, 1).cuda().float()  )
                              
        if mode == 1:
            print(" Use the estimated LPS.")
            recovered_lps = noisy_htkdata +  np.log( torch.squeeze( irm_outputs[stage_select-1]).cpu().data.numpy())   
        elif mode == 2:
            print(" Use the estimated IRM.")
            recovered_lps =  torch.squeeze( lps_outputs[stage_select-1]).cpu().data.numpy()  * global_var[:257] +  global_mean[:257] 
        elif mode == 3:
            print(" Use the fusion of estimated LPS and IRM.")
            recovered_lps = 0.5*( noisy_htkdata +  np.log( torch.squeeze( irm_outputs[stage_select-1]).cpu().data.numpy() )  ) + 0.5*( torch.squeeze( lps_outputs[stage_select-1]).cpu().data.numpy()  * global_var[:257] +  global_mean[:257]  )          
 
        # Reconstruct audio.
        wave_recon = utils.logspec2wav(
            recovered_lps, temp, window=np.hamming(WL), n_per_seg=WL,
            noverlap=WL2)
        data_se.append(wave_recon)

    data_se = [x.astype(np.int16, copy=False) for x in data_se]
    data_se = np.concatenate(data_se)
    wav_io.write(dest_wav_file, SR, data_se)



def main_denoising(wav_files, output_dir, verbose, use_gpu, truncate_minutes, mode,stage_select=3):
    """Perform speech enhancement for WAV files in ``wav_dir``.

    Parameters
    ----------
    wav_files : list of str
        Paths to WAV files to enhance.

    output_dir : str
        Path to output directory for enhanced WAV files.

    verbose : bool, optional
        If True, print full stacktrace to STDERR for files with errors.

    kwargs
        Keyword arguments to pass to ``denoise_wav``.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load global MVN statistics.
    global_mean = np.load(os.path.join(HERE, 'mean.npy') )
    global_var = 1 / np.load(os.path.join(HERE, 'inv_std.npy') )

    # Perform speech enhancement.
    for src_wav_file in wav_files:
        # Perform basic checks of input WAV.
        if not os.path.exists(src_wav_file):
            utils.error('File "%s" does not exist. Skipping.' % src_wav_file)
            continue
        if not utils.is_wav(src_wav_file):
            utils.error('File "%s" is not WAV. Skipping.' % src_wav_file)
            continue
        if utils.get_sr(src_wav_file) != SR:
            utils.error('Sample rate of file "%s" is not %d Hz. Skipping.' %
                        (src_wav_file, SR))
            continue
        if utils.get_num_channels(src_wav_file) != NUM_CHANNELS:
            utils.error('File "%s" is not monochannel. Skipping.' % src_wav_file)
            continue
        if utils.get_bitdepth(src_wav_file) != BITDEPTH:
            utils.error('Bitdepth of file "%s" is not %d. Skipping.' %
                        (src_wav_file, BITDEPTH))
            continue

        # Denoise.
        try:
            bn = os.path.basename(src_wav_file)
            dest_wav_file = os.path.join(output_dir, bn)
            denoise_wav(src_wav_file, dest_wav_file, global_mean, global_var, use_gpu, truncate_minutes, mode, stage_select )
            print('Finished processing file "%s".' % src_wav_file)
        except Exception as e:
            msg = 'Problem encountered while processing file "%s". Skipping.' % src_wav_file
            if verbose:
                msg = '%s Full error output:\n%s' % (msg, e)
            utils.error(msg)
            continue


# TODO: Logging is getting complicated. Consider adding a custom logger...
def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description='Denoise WAV files.', add_help=True)
    parser.add_argument(
        '--wav_dir', nargs=None, type=str, metavar='STR',
        help='directory containing WAV files to denoise '
             '(default: %(default)s')
    parser.add_argument(
        '--output_dir', nargs=None, type=str, metavar='STR',
        help='output directory for denoised WAV files (default: %(default)s)')
    parser.add_argument(
        '-S', dest='scpf', nargs=None, type=str, metavar='STR',
        help='script file of paths to WAV files to denosie (detault: %(default)s)')
    parser.add_argument(
        '--use_gpu', nargs=None, default='true', type=str, metavar='STR',
        choices=['true', 'false'],
        help='whether or not to use GPU (default: %(default)s)')
    parser.add_argument(
        '--truncate_minutes', nargs=None, default=10, type=float,
        metavar='FLOAT',
        help='maximum chunk size in minutes (default: %(default)s)')
    parser.add_argument(
        '--mode', nargs=None, default=3, type=float,
        metavar='INT',
        help='which output to use: (1:irm , 2:lps, 3:fusion) (default: %(default)s)')     
    parser.add_argument(
        '--stage_select', nargs=None, default=3, type=int,
        metavar='INT',
        help='which stage(1 or 2 or 3) of PL based model, only works for "1000h model" (default: %(default)s)') 
    parser.add_argument(
        '--verbose', default=False, action='store_true',
        help='print full stacktrace for files with errors')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if not utils.xor(args.wav_dir, args.scpf):
        parser.error('Exactly one of --wav_dir and -S must be set.')
        sys.exit(1)
    use_gpu = args.use_gpu == 'true'

    # Determine files to denoise.
    if args.scpf is not None:
        wav_files = utils.load_script_file(args.scpf, '.wav')
    else:
        wav_files = utils.listdir(args.wav_dir, ext='.wav')

    # Determine output directory for denoised audio.
    if args.output_dir is None and args.wav_dir is not None:
        utils.warn('Output directory not specified. Defaulting to "%s"' %
                   args.wav_dir)
        args.output_dir = args.wav_dir

    # Perform denoising.
    main_denoising(
        wav_files, args.output_dir, args.verbose, use_gpu=use_gpu,
        truncate_minutes=args.truncate_minutes, mode=args.mode, stage_select=args.stage_select )

#def main_denoising(wav_files, output_dir, verbose=False, **kwargs):

if __name__ == '__main__':
    main()
