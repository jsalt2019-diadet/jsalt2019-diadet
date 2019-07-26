#!/usr/bin/env python3
#
#               
#
# Apache 2.0
#
# 
#  -- Modified from Zili Huang's (Johns Hopkins University) VB_resegmentation.py 
#  -- Uses the speaker attribution (q) matrix from VB resegmentation and an overlap 
#  -- hypothesis rttm to assign speakers to overlapped frames 
#
# JSALT 2019, Latané Bullock

import numpy as np
import argparse


def get_utt_list(utt2spk_filename):
    with open(utt2spk_filename, 'r') as fh:
        content = fh.readlines()
    utt_list = [line.split()[0] for line in content]
    print("{} utterances in total".format(len(utt_list)))
    return utt_list

# prepare utt2num_frames dictionary
def get_utt2num_frames(utt2num_frames_filename):
    utt2num_frames = {}
    with open(utt2num_frames_filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt2num_frames[line_split[0]] = int(line_split[1])
    return utt2num_frames

def rttm2one_hot(uttname, utt2num_frames, full_rttm_filename):
    num_frames = utt2num_frames[uttname]

    ref = np.zeros(num_frames)

    with open(full_rttm_filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        uttname_line = line_split[1]
        if uttname != uttname_line:
            continue
        start_time, duration = int(float(line_split[3]) * 100), int(float(line_split[4]) * 100)
        end_time = start_time + duration
        
        for i in range(start_time, end_time):
            if i < 0:
                raise ValueError("Time index less than 0")
            elif i >= num_frames:
                print('rttm extends beyond the number of frames...')
                print(line)
                print('Start time: ', start_time)
                print('End time: ', end_time)
                print('i: ', i) 
                print('num frame: ', num_frames)               
                # raise ValueError("Time index exceeds number of frames")
                break
            else:
                ref[i] = 1
    return ref.astype(int)

# create output rttm file
def create_rttm_output(uttname, pri_sec, predicted_label, output_dir, channel):
    num_frames = len(predicted_label)

    start_idx = 0
    seg_list = []

    last_label = predicted_label[0]
    for i in range(num_frames):
        if predicted_label[i] == last_label: # The speaker label remains the same.
            continue
        else: # The speaker label is different.
            if last_label != 0: # Ignore the silence.
                seg_list.append([start_idx, i, last_label])
            start_idx = i
            last_label = predicted_label[i]
    if last_label != 0:
        seg_list.append([start_idx, num_frames, last_label])

    with open("{}/tmp/{}_predict_{}.rttm".format(output_dir, uttname, pri_sec), 'w') as fh:
        for i in range(len(seg_list)):
            start_frame = (seg_list[i])[0]
            end_frame = (seg_list[i])[1]
            label = (seg_list[i])[2]
            duration = end_frame - start_frame
            fh.write("SPEAKER {} {} {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(uttname, channel, start_frame / 100.0, duration / 100.0, label))
    return 0



def main():
    parser = argparse.ArgumentParser(description='Frame-level overlap reassignment with speaker posterior attributions')
    parser.add_argument('ovl_dir', type=str, help='Path to directory where we have necessary files for overlap reassignment')

    args = parser.parse_args()
    print(args)

    utt_list = get_utt_list("{}/utt2spk".format(args.ovl_dir))
    utt2num_frames = get_utt2num_frames("{}/utt2num_frames".format(args.ovl_dir))
    
    for utt in utt_list:
        n_frames = utt2num_frames[utt]

        vad = rttm2one_hot(utt, utt2num_frames, '{}/rttm_in'.format(args.ovl_dir))
        # unique, counts = np.unique(vad, return_counts=True)
        # voiced_frames = dict(zip(unique, counts))[1]

        overlap = rttm2one_hot(utt, utt2num_frames, '{}/overlap.rttm'.format(args.ovl_dir))
       
        # Keep only the voiced frames (0 denotes the silence 
        # frames, 1 denotes the overlapping speech frames).
        mask = (vad >= 1)

        # Remember: q is only for voiced frames
        q_out = np.load('{}/q_mats/{}_q_out.npy'.format(args.ovl_dir, utt))
        
        # Standard procedure from VB reseg - take the most likely speaker 
        predicted_label_voiced = np.argsort(-q_out, 1)[:,0] + 2 
        predicted_label = (np.zeros(len(mask))).astype(int)
        predicted_label[mask] = predicted_label_voiced
        # This is the 'primary' speaker for each frame
        create_rttm_output(utt, 'pri', predicted_label, args.ovl_dir, channel=1)

        # Write "secondary" speakers for overlap regions 
        # -- take second most likely speaker for each overlap frame 
        predicted_label_voiced = np.argsort(-q_out, 1)[:,1] + 2 
        predicted_label = (np.zeros(len(mask))).astype(int)

        frame_t_voiced = 0
        for t in range(len(mask)):
            if vad[t] >= 1:
                if overlap[t] >= 1:
                    predicted_label[t] = predicted_label_voiced[frame_t_voiced]
                frame_t_voiced += 1
            
        create_rttm_output(utt, 'sec', predicted_label, args.ovl_dir, channel=1)


    return 0



if __name__ == "__main__":
    main()
