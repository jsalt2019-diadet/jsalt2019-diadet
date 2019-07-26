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
# Diego Castan

from optparse import OptionParser
import numpy as np

def main():
    usage = "%prog [options] txtfile outputdir"
    desc = "Convert the txtfile from diarization of the from: \
            ID t_in t_out \
            into a kaldi format file for spkdet task"
    version = "%prog 0.1"
    parser = OptionParser(usage=usage, description=desc, version=version)
    parser.add_option("-o", "--outputname", action="store", type="string", help="Output file name", default=None)
    (opt, args) = parser.parse_args()

    if(len(args)!=2):
        parser.error("Incorrect number of arguments")
    inputfile, outputdir = args

    # Read document and loaded in memory
    fr = open(inputfile)
    lines = fr.readlines()
    fr.close()

    if opt.outputname:
        fws = open(outputdir+'/'+'segoverlap_' +opt.outputname,'wt')
        fwrttm = open(outputdir+'/'+ opt.outputname + '.rttm','wt')
    else:
        fws = open(outputdir+'/segoverlap','wt')
        fwrttm = open(outputdir+'/overlap.rttm','wt')

    # Process each line
    for i,linea in enumerate(lines):
        name, tin, tout = linea.strip().split()
        frmin = float(tin)*100.0
        frmout = float(tout)*100.0
        segmin = int(np.floor(frmin/6000.0))*6000
        segmout = segmin+6000
        tsin = frmin%6000.0
        tsout = frmout%6000.0
        namef1 = name + '-' + format(segmin, '07') + '-' +format(segmout, '07') + '-' +format(int(tsin), '07') + '-' +format(int(tsout), '07')
        namef2 = name + '-' + format(segmin, '07') + '-' +format(segmout, '07')
        tfin = float(tsin/100.0)
        tfout = float(tsout/100.0)
        fws.write('%s %s %s %s\n' % (namef1,namef2,format(tfin, '.2f'),format(tfout, '.2f')))
        # print name, tin, tout
        # print namef1, namef2, tfin, tfout
        fwrttm.write('SPEAKER %s 1 %s %s <NA> <NA> speech 1 <NA>\n' % (namef2,format(tfin, '.2f'),format(tfout-tfin, '.2f')))


if __name__=="__main__":
    main()
