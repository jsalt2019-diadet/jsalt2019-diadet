#!/usr/bin/env python
"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""

import sys
import os
import argparse
import time
import logging
import subprocess
import re

import numpy as np
import pandas as pd

def find_audios(wav_path):

    command = 'find %s -name "*.wav"' % (wav_path)
    wavs = subprocess.check_output(command, shell=True).decode('utf-8').splitlines()
    keys = [ os.path.splitext(os.path.basename(wav))[0] for wav in wavs ]
    data = {'key': keys, 'file_path': wavs}
    df_wav = pd.DataFrame(data)
    return df_wav


def read_rttm(list_path):

    rttm_file='%s/all.rttm' % (list_path)
    rttm = pd.read_csv(rttm_file, sep='\t', header=None,
                       names=['segment_type','file_id','chnl','tbeg','tdur',
                              'ortho','stype','name','conf','slat'])
    #remove empty lines:
    index = (rttm['tdur']>= 0.025)
    rttm = rttm[index]
    rttm['stype'] = '<NA>'
    return rttm


def remove_overlap_from_rttm_vad(rttm):

    tbeg_index = rttm.columns.get_indexer(['tbeg'])
    tdur_index = rttm.columns.get_indexer(['tdur'])
    tend = np.asarray(rttm['tbeg'] + rttm['tdur'])
    index = np.ones(rttm.shape[0], dtype=bool)
    p = 0
    for i in range(1, rttm.shape[0]):
        if rttm['file_id'].iloc[p] == rttm['file_id'].iloc[i]:
            if tend[p] > rttm.iloc[i, tbeg_index].item():
                index[i] = False
                tend[p] = tend[i]
                new_dur = tend[i] - rttm.iloc[p, tbeg_index].item()
                rttm.iloc[p, tdur_index] = new_dur
            else:
                p = i
        else:
            p = i

    rttm = rttm.loc[index]
    return rttm
            

def filter_wavs(df_wav, file_names):
    df_wav = df_wav.loc[df_wav['key'].isin(file_names)].sort_values('key')
    return df_wav


def write_wav(df_wav, output_path):

    with open(output_path + '/wav.scp', 'w') as f:
        for key,file_path in zip(df_wav['key'], df_wav['file_path']):
            f.write('%s %s\n' % (key, file_path))


def write_dummy_utt2spk(file_names, output_path):
    
    with open(output_path + '/utt2spk', 'w') as f:
        for fn in file_names:
            f.write('%s %s\n' % (fn, fn))


def write_rttm_vad(df_vad, output_path):
    file_path = output_path + '/vad.rttm'
    df_vad[['segment_type', 'file_id', 'chnl',
            'tbeg','tdur','ortho', 'stype',
            'name', 'conf', 'slat']].to_csv(
                file_path, sep=' ', float_format='%.3f',
                index=False, header=False)


def write_rttm_spk(df_vad, output_path):
    file_path = output_path + '/diarization.rttm'
    df_vad[['segment_type', 'file_id', 'chnl',
            'tbeg','tdur','ortho', 'stype',
            'name', 'conf', 'slat']].to_csv(
                file_path, sep=' ', float_format='%.3f',
                index=False, header=False)



def write_vad_segm_fmt(rttm_vad, output_path):
    with open(output_path + '/vad.segments', 'w') as f:
        for row in rttm_vad.itertuples():
            tbeg = row.tbeg
            tend = row.tbeg + row.tdur
            segment_id = '%s-%07d-%07d' % (row.file_id, int(tbeg*100), int(tend*100))
            f.write('%s %s %.2f %.2f\n' % (segment_id, row.file_id, tbeg, tend))

    

def make_jsalt19_spkdiar(list_path, wav_path, output_path, data_name, partition):

    output_path = '%s/%s_%s' % (output_path, data_name, partition)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print('read audios')
    df_wav = find_audios(wav_path)
    print('read rttm')
    rttm = read_rttm(list_path)

    print('make wav.scp')
    file_names = rttm['file_id'].sort_values().unique()
    df_wav = filter_wavs(df_wav, file_names)
    write_wav(df_wav, output_path)

    print('write utt2spk')
    write_dummy_utt2spk(file_names, output_path)

    print('write diar rttm')
    write_rttm_spk(rttm, output_path)

    #create vad rttm
    print('make vad rttm')
    rttm_vad = rttm.copy()
    rttm_vad['name'] = 'speech'
    rttm_vad = remove_overlap_from_rttm_vad(rttm_vad)
    write_rttm_vad(rttm_vad, output_path)

    #write vad in segment format
    print('write vad segments')
    write_vad_segm_fmt(rttm_vad, output_path)
    

    
    

if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Make JSALT19 datasets for spk diarization')

    parser.add_argument('--list-path', dest='list_path', required=True)
    parser.add_argument('--wav-path', dest='wav_path', required=True)
    parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument('--data-name', dest='data_name', required=True)
    parser.add_argument('--partition', dest='partition', choices=['train', 'dev', 'eval'], required=True)
    args=parser.parse_args()
    
    make_jsalt19_spkdiar(**vars(args))
