#!/usr/bin/env python3

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

import bin.getlc as gl
import bin.lctips as lt
import bin.fft as ft


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
            if os.path.isfile(fkey+"_k2.dat"):
                lck2    = np.loadtxt(fkey+"_k2.dat", dtype='f8').T
                lctess  = np.loadtxt(fkey+"_tess.dat", dtype='f8').T
            else:
                lck2    = gl.k2lc_byepic(k2id)
                print("downloaded k2 data.")
                lctess  = gl.tesslc_byepic(k2id)
                if len(lctess) != 1:
                    print("downloaded tess data.")
                    np.savetxt("lightcurves/"+k2id+"_k2.dat",lck2.T)
                    np.savetxt("lightcurves/"+k2id+"_tess.dat",lctess.T)
            
            i+=1