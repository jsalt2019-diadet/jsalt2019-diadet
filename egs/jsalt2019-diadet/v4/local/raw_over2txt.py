#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2019 CNRS

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

from typing import TextIO
from typing import Union
from pyannote.core import Timeline
from pyannote.core import Annotation
from optparse import OptionParser
from pyannote.database import get_protocol
from pyannote.audio.features import Precomputed
from pyannote.audio.signal import Binarize


def write_txt(file: TextIO, output: Union[Timeline, Annotation]):
    """Write pipeline output to "txt" file

    Parameters
    ----------
    file : file object
    output : `pyannote.core.Timeline` or `pyannote.core.Annotation`
        Pipeline output
    """


    if isinstance(output, Timeline):
        for s in output:
            line = f'{output.uri} {s.start:.3f} {s.end:.3f}\n'
            file.write(line)
        return


    if isinstance(output, Annotation):
        for s, t, l in output.itertracks(yield_label=True):
            line = f'{output.uri} {s.start:.3f} {s.end:.3f} {t} {l}\n'
            file.write(line)
        return


    msg = (
        f'Dumping {output.__class__.__name__} instances to "txt" files '
        f'is not supported.'
    )
    raise NotImplementedError(msg)

def main():
    usage = "%prog [options] database, raw_score_path"
    desc = "Write the output of the binary overlap detector into test based on a threshold"
    version = "%prog 0.1"
    parser = OptionParser(usage=usage, description=desc, version=version)
    parser.add_option("-t", "--onset", action="store", type="float", help="Onset Threshold", default=0.70)
    parser.add_option("-f", "--offset", action="store", type="float", help="Offset Threshold", default=0.70)
    parser.add_option("-d", "--dev", action="store_true", help="Print output based on development set", default=False)
    parser.add_option("-o", "--outputfile", action="store", type="string", help="Output file", default="./overlap.txt")
    (opt, args) = parser.parse_args()

    if(len(args)!=2):
        parser.error("Incorrect number of arguments")
    database, raw_score_path = args

    # get test file of protocol
    protocol = get_protocol(database)

    # load precomputed overlap scores as pyannote.core.SlidingWindowFeature
    precomputed = Precomputed(raw_score_path)
    # StackedRNN model
    # initialize binarizer
    # onset / offset are tunable parameters (and should be tuned for better 
    # performance). we use log_scale=True because of the final log-softmax in the 
    binarize = Binarize(onset=opt.onset, offset=opt.offset, log_scale=True)

    fw = open(opt.outputfile, 'a+')

    if opt.dev:
        for test_file in protocol.development():
            ovl_scores = precomputed(test_file)


            # binarize overlap scores to obtain overlap regions as pyannote.core.Timeline
            ovl_regions = binarize.apply(ovl_scores, dimension=1)
            ovl_regions.uri = test_file['uri']


            # write the output into text
            write_txt(fw, ovl_regions)
 
    else:
        for test_file in protocol.test():
            ovl_scores = precomputed(test_file)


            # binarize overlap scores to obtain overlap regions as pyannote.core.Timeline
            ovl_regions = binarize.apply(ovl_scores, dimension=1)
            ovl_regions.uri = test_file['uri']


            # write the output into text
            write_txt(fw, ovl_regions)
    fw.close()


if __name__=="__main__":
    main()
