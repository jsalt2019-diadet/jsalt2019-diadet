#!/usr/bin/env python3

from __future__ import print_function
import os
import sys
import argparse

from collections import OrderedDict
from scipy.io import wavfile
from numpy import linalg as LA
import numpy as np

def get_args():
    parser = argparse.ArgumentParser('This script maps each simulated utterances to '
                                    'its corresponding rt60')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('rir2rt60_map', type=str)
    parser.add_argument('utt2reverbinfo_file', type=str)

    args = parser.parse_args()

    return args


def check_args(args):
    if not os.path.isdir(args.data_dir):
        raise ValueError('inp data dir {d} does not exist'.format(d=args.data_dir))
    args.wav_scp = '{d}/wav.scp'.format(d=args.data_dir)
    args.utt2uniq = '{d}/utt2uniq'.format(d=args.data_dir)

    if not os.path.isfile(args.wav_scp):
        raise ValueError('File provided wav scp {w} does not exist'.format(
                        w=args.wav_scp))
    if not os.path.isfile(args.utt2uniq):
        raise ValueError('File provided utt2uniq {w} does not exist'.format(
                        w=args.utt2uniq))
    if not os.path.isfile(args.rir2rt60_map):
        raise ValueError('File provided rt60 info file {r} does not exist'.format(
                        r=args.rir2rt60_map))

    return args


def map_rirs_to_rt60s(ifile):
    with open(ifile) as f:
        content = f.read().splitlines()

    ririd_to_rt60 = {}
    for line in content:
        line_parsed = line.strip().split()
        ririd, roomid, rt60 = line_parsed[0], line_parsed[1], line_parsed[2]
        ririd_to_rt60[ririd] = rt60

    return ririd_to_rt60

def map_utt_to_uniq(ifile):
    with open(ifile) as f:
        content = f.read().splitlines()

    utt2uniq = {}
    for line in content:
        line_parsed = line.strip().split()
        utt, uniq = line_parsed[0], line_parsed[1]
        utt2uniq[utt] = uniq

    return utt2uniq


def map_utts_to_room_ids(ifile):
    # 100304-sre06-kacg-a-reverb sph2pipe -f wav -p -c 1 /export/corpora/LDC/LDC2011S09/data/train/data/kacg.sph | wav-reverberate --shift-output=true --impulse-response="sox RIRS_NOISES/simulated_rirs/smallroom/Room200/Room200-00049.wav -r 8000 -t wav - |" - - |
    with open(ifile) as f:
        content = f.read().splitlines()

    #utt_to_room_id = {}
    utt_to_room_info = OrderedDict()
    for line in content:
        line_parsed = line.strip().split()
        utt = line_parsed[0]
        utt_to_room_info[utt] = {}
        for key in line_parsed[1:]:
            if "RIRS_NOISES" in key:
                rir = key
                utt_to_room_info[utt]['rir'] = rir
                break

        rir_parsed = rir.split('/')
        room_type = rir_parsed[2]
        room_id = rir_parsed[-2]
        rir_id = rir_parsed[-1].split('.wav')[0]

        if 'small' in room_type:
            kwrd = 'small'
            utt_to_room_info[utt]['roomtype'] = 'smallroom'
        elif 'medium' in room_type:
            kwrd = 'medium'
            utt_to_room_info[utt]['roomtype'] = 'mediumroom'
        elif 'large' in room_type:
            kwrd = 'large'
            utt_to_room_info[utt]['roomtype'] = 'largeroom'
        else:
            raise ValueError('unknown room type {r} found'.format(r=room_id))

        utt_to_room_info[utt]['roomid'] = kwrd + '-' + room_id
        utt_to_room_info[utt]['ririd'] = kwrd + '-' + rir_id

    return utt_to_room_info


def get_h_n_direct_and_n_direct_from_rir(rir, normalize_rir=True):
    fs, data = wavfile.read(rir)
    if normalize_rir:
        data = data/LA.norm(data)

    return np.max(data), np.argmax(data)

def main():
    args = get_args()
    args = check_args(args)

    ririd_to_rt60 = map_rirs_to_rt60s(args.rir2rt60_map)
    utts_to_roominfo = map_utts_to_room_ids(args.wav_scp)
    utt2uniq = map_utt_to_uniq(args.utt2uniq)

    print('\nCreating utt2reverbinfo file {i}'.format(i=args.utt2reverbinfo_file))

    with open(args.utt2reverbinfo_file, 'w') as f:
        for utt in utts_to_roominfo:
            roomid = utts_to_roominfo[utt]['roomid']
            ririd = utts_to_roominfo[utt]['ririd']
            roomtype = utts_to_roominfo[utt]['roomtype']
            rt60 = ririd_to_rt60[ririd]
            rir = utts_to_roominfo[utt]['rir']
            uniq = utt2uniq[utt]
            h_n_direct, n_direct = get_h_n_direct_and_n_direct_from_rir(rir)
            #f.write('{utt} {rt} {rid}\n'.format(utt=utt, rt=rt60, rid=roomid))
            f.write('{utt} {uniq} {roomid} {rt} {h_n} {n}\n'.format(
                        utt=utt, roomid=roomid, rt=rt60, uniq=uniq, h_n=h_n_direct, n=n_direct))

    print('Successfully created utt2reverbinfo file: {f}'.format(f=args.utt2reverbinfo_file))


if __name__ == '__main__':
    main()
