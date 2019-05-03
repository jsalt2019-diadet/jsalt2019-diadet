#! /usr/bin/python      
import os
import sys


rttm = sys.argv[1]
seg = sys.argv[2] + "/segments"
utt = sys.argv[2] + "/utt2spk"
tmp = "/export/c03/fwu/dihard/baby_train/rttm2seg.tmp"

utt_ID = os.path.basename(rttm)
utt_ID = os.path.splitext(utt_ID)[0]

wf_tmp = open(tmp,"w+")
wf_seg = open(seg,"a")
wf_utt = open(utt,"a")

seg_cnt = 0
with open(rttm) as fp:
    line = fp.readline() 
    while line:
        seg_cnt += 1
        toks = line.strip().split()
        
        start_t = float(toks[3])
        duration = float(toks[4])  
        if duration == 0.0:
            line = fp.readline()
            continue

        end_t = start_t + duration
        
        # spk_ID = toks[7]
        # spk_ID = utt_ID + "-" + spk_ID
        
        seg_ID = utt_ID + "-" +str(seg_cnt).zfill(4)
        seg_line = [seg_ID,utt_ID,str(start_t),str(end_t),'\n']

        utt_line = [seg_ID,utt_ID,'\n']
        
        # Write segments and utt2spk
        wf_seg.write(" ".join(seg_line))
        wf_tmp.write(" ".join(seg_line))
        wf_utt.write(" ".join(utt_line))
            
        line = fp.readline()
fp.close()

wf_tmp.close()
wf_seg.close()
wf_utt.close()
    

