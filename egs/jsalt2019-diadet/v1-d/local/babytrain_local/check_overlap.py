#! /bin/sur/python

import os
import sys

def file_overlap(segs):
    sorted_start = sorted(segs, key=lambda x: x[0])

    overlap = 0
    duration = 0

    num_segs = len(segs)
    for i in range(num_segs - 1):
        # Overlap
        cur_end = sorted_start[i][1]
        nxt_start = sorted_start[i+1][0]

        if cur_end > nxt_start:
            overlap += cur_end - nxt_start

    return overlap,file_duration


dir = sys.argv[1]
overlap_log = sys.argv[2]

total_duration = 0.0
overlap_duration = 0.0


for file in os.listdir(dir):
    name = os.path.basename(file)
    if name[-5:] == ".rttm":
        segs = []
        with open(dir + '/' + file,"r") as fp:
            line = fp.readline()
            while line:
                toks = line.strip().split()                
                start = float(toks[3])
                duration = float(toks[4])
                
                # Ignore 0 duration
                if duration == 0.0:
                    line = fp.readline()
                    continue
                else:
                    segs.append((start,start+duration))

                line = fp.readline()
        fp.close()

        aud_overlap, aud_length = file_overlap(segs)
        total_duration += aud_length
        overlap_duration += aud_overlap

duration_hr = total_duration / 3600.0
overlap_hr = overlap_duration / 3600.0


result =["Total duration: ",str(duration_hr),"    Overlap duration: ",str(overlap_hr),"\n"]
F = open(overlap_log, "a")
res_line = ''.join(result)
F.write(res_line)
# print("Total duration:",duration_hr,"    Overlap duration:",overlap_hr)
