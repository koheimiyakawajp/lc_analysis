#!/usr/bin/env python3

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

import bin.getlc as gl
import bin.lctips as lt
import bin.fft as ft


def dlwrap(fkey, mkey,k2id):
    if os.path.isfile(fkey+"_"+mkey+".dat"):
        print(fkey+"_"+mkey+".dat")
        print("light curve already exists.")
    else:
        print("downloading "+mkey + " light curve.")
        if mkey=="k2":
            lc    = gl.k2lc_byepic(k2id)
        elif mkey=="tess":
            lc    = gl.tesslc_byepic(k2id)
        elif mkey=="tess_qlp":
            lc    = gl.tesslcQLP_byepic(k2id)
        print("done.")

        if len(lc) != 1:
            print("save lc data.")
            np.savetxt(fkey+"_"+mkey+".dat",lc.T)

if __name__=='__main__':
    fname   = sys.argv[1]
    dlist   = np.loadtxt(fname, dtype='unicode',comments='#')
    epiclist    = dlist.T

    i = 0
    out_array   = []
    for k2id in epiclist:
        tid     = gl.EPIC_to_TIC(k2id)
        if tid != -1:
            print("EPIC "+k2id, tid)
            fkey    = "lightcurves/"+k2id
            dlwrap(fkey, "k2", k2id) 
            dlwrap(fkey, "tess", k2id) 
            dlwrap(fkey, "tess_qlp", k2id) 
            i+=1