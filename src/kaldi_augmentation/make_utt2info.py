#!/usr/bin/env python3

from __future__ import print_function
import os
import sys
import argparse

from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser('This script maps each simulated utterances to '
                                    'its corresponding rt60')
    parser.add_argument('--utt2snr-file', type=str, required=True,
                                        help='utt2snr file')
    parser.add_argument('--utt2reverb-file', type=str, required=True,
                                        help='utt2snr file')
    parser.add_argument('--utt2info-file', type=str, required=True,
                                        help='utt2info file')

    # optional args
    parser.add_argument('--additive-noise-types', nargs='+', type=str,
                                default=['babble', 'music', 'noise'])

    args = parser.parse_args()

    return args


def check_args(args):
    if not os.path.isfile(args.utt2snr_file):
        raise ValueError('inp utt2snr file {d} does not exist'.format(
                                    d=args.utt2snr_file))
    if not os.path.isfile(args.utt2reverb_file):
        raise ValueError('inp utt2snr file {d} does not exist'.format(
                                    d=args.utt2reverb_file))

    if not os.path.isdir(os.path.dirname(args.utt2info_file)):
        os.makedirs(os.path.dirname(args.utt2info_file))
    return args

def read_utt2snr_file(ifile, aug_types):
    print(aug_types)
    with open(ifile) as f:
        content = f.read().splitlines()

    utt2uniq_and_snr = defaultdict(dict)
    for line in content:
        line_parsed = line.strip().split()
        utt, uniq_utt, snr = line_parsed
        utt2uniq_and_snr[utt]['uniq'] = uniq_utt
        utt2uniq_and_snr[utt]['snr'] = snr
        for aug in aug_types:
            if aug in utt:
                utt2uniq_and_snr[utt]['type'] = aug
                break

    return utt2uniq_and_snr

def read_utt2reverbinfo_file(ifile):
    with open(ifile) as f:
        content = f.read().splitlines()

    utt2reverb_info = defaultdict(dict)
    for line in content:
        line_parsed = line.strip().split()
        utt, uniq_utt, roomid, rt60, h_n_index, n_index = line_parsed
        utt2reverb_info[utt]['uniq'] = uniq_utt
        utt2reverb_info[utt]['roomid'] = roomid
        utt2reverb_info[utt]['rt60'] = rt60
        utt2reverb_info[utt]['h_n_index'] = h_n_index
        utt2reverb_info[utt]['n_index'] = n_index

    return utt2reverb_info

def main():
    args = get_args()
    args = check_args(args)

    utt2snr_info = read_utt2snr_file(args.utt2snr_file, args.additive_noise_types)
    utt2reverb_info = read_utt2reverbinfo_file(args.utt2reverb_file)

    print('\nCreating utt2info file {i}'.format(i=args.utt2info_file))

    with open(args.utt2info_file, 'w') as f:
        # First create for additive noise directories
        for utt in utt2snr_info:
            uniq_utt = utt2snr_info[utt]['uniq']
            aug_type = utt2snr_info[utt]['type']
            snr = utt2snr_info[utt]['snr']

            string = '{u} {un} {a} {s} NA NA NA NA'.format(
                        u=utt, un=uniq_utt, a=aug_type, s=snr)
            f.write(string + '\n')

        for utt in utt2reverb_info:
            uniq_utt = utt2reverb_info[utt]['uniq']
            roomid = utt2reverb_info[utt]['roomid']
            rt60 = utt2reverb_info[utt]['rt60']
            h_n_index = utt2reverb_info[utt]['h_n_index']
            n_index = utt2reverb_info[utt]['n_index']
            string = '{u} {un} NA NA {ri} {rt60} {h_n} {n}'.format(
                        u=utt, un=uniq_utt, ri=roomid, rt60=rt60,
                                            h_n=h_n_index,n=n_index)
            f.write(string + '\n')



    print('Successfully created utt2info file: {f}'.format(f=args.utt2info_file))


if __name__ == '__main__':
    main()
