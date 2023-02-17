#!/usr/bin/env python3

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

import bin.getlc as gl
import bin.lctips as lt
import bin.fft as ft
vfile   = "lightcurves/vaclist.dat"

def dlwrap(fkey, mkey,k2id, vaclist):
    if np.any(vaclist==k2id+mkey):
        print("light curve does not exist on remote server.")
        return 1
    elif os.path.isfile(fkey+"_"+mkey+".dat"):
        print(fkey+"_"+mkey+".dat")
        print("light curve already exists.")
        return 2
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
            return 0
        else:
            vaclist.append(k2id+mkey)
            return 1

if __name__=='__main__':
    fname   = sys.argv[1]
    dlist   = np.loadtxt(fname, dtype='unicode',comments='#')
    epiclist    = dlist.T
    
    if os.path.isfile(vfile):
        vaclist     = np.loadtxt(vfile, dtype='unicode').T
    else:
        vaclist     = []


    i = 0
    out_array   = []
    for k2id in epiclist:
        tid     = gl.EPIC_to_TIC(k2id)
        if tid != -1:
            print("EPIC "+k2id, tid)
            fkey    = "lightcurves/"+k2id
            flg     = dlwrap(fkey, "k2", k2id, vaclist) 
            if flg != 1:
                dlwrap(fkey, "tess", k2id, vaclist) 
                dlwrap(fkey, "tess_qlp", k2id, vaclist) 
            i+=1
        else:
            print("EPIC " + k2id +" does not match any TICs.")
    
    vaclist     = np.array(vaclist)    
    np.savetxt(vfile, vaclist.reshape(-1,1), fmt="%s")