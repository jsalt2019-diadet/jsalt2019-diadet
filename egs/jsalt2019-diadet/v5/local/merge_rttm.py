#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2019 JSALT

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr
# Diego Castan

import pyannote.database
from pyannote.core import Annotation, Timeline, Segment, SlidingWindow
from typing import TextIO
from typing import Union
from optparse import OptionParser

def write_rttm(file: TextIO,	output: Union[Timeline, Annotation], label=None):
    """Write pipeline output to "rttm" file
    Parameters
    ----------
    file : file object
    output : `pyannote.core.Timeline` or `pyannote.core.Annotation`
	Pipeline output
    """

    if isinstance(output, Timeline):
        output = output.to_annotation(generator='string')

    if isinstance(output, Annotation):
        for s, t, l in output.itertracks(yield_label=True):
            if label:
                line = (
                    f'SPEAKER {output.uri} 1 {s.start:.3f} {s.duration:.3f} '
                    f'<NA> <NA> {label} <NA> <NA>\n'
                )
            else:
                line = (
                    f'SPEAKER {output.uri} 1 {s.start:.3f} {s.duration:.3f} '
                    f'<NA> <NA> {l} <NA> <NA>\n'
                )
            file.write(line)
        return

    msg = (
	f'Dumping {output.__class__.__name__} instances to "rttm" files '
	f'is not supported.'
    )
    raise NotImplementedError(msg)

def main():
    usage = "%prog [options] RTTMone RTTMtwo"
    desc = "Convert the txtfile from diarization of the from: \
            ID t_in t_out \
            into a kaldi format file for spkdet task"
    version = "%prog 0.1"
    parser = OptionParser(usage=usage, description=desc, version=version)
    (opt, args) = parser.parse_args()

    if(len(args)!=3):
        parser.error("Incorrect number of arguments")
    vadrttm, overlaprttm, outputrttm = args

    # Read document and loaded in memory
    vad = pyannote.database.util.load_rttm(vadrttm)
    ovl = pyannote.database.util.load_rttm(overlaprttm)

    fw = open(outputrttm,'wt')
    for name in vad:
        
        # Examples
        # speech = vad['EN2002a.Mix-Headset-0000000-0006000'].get_timeline()
        # duration = vad['EN2002a.Mix-Headset-0000000-0006000'].get_timeline()[-1][1]
        # overlap = ovl['EN2002a.Mix-Headset-0000000-0006000'].get_timeline()
        speech = vad[name].get_timeline()
        duration = vad[name].get_timeline()[-1][1]
        if name in ovl.keys():
            overlap = ovl[name].get_timeline()

            # just get the intersections of the VAD and overlap
            intersection = Timeline()
            for speech_segment, overlap_segment in speech.co_iter(overlap):
                intersection.add(speech_segment & overlap_segment)

            keep = intersection.gaps(support=Segment(0, duration))

            vad_without_overlap = speech.crop(keep)
        else:
            vad_without_overlap = speech

        # Write RTTM
        write_rttm(fw, vad_without_overlap, label='speech')
    fw.close()
    
if __name__=="__main__":
    main()
