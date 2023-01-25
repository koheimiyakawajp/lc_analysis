#!/usr/bin/env python3

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

import bin.getlc as gl
import bin.lctips as lt
import bin.fft as ft

def plotfunc(lck2 ,lck2_1, lck2_nn, lctess, lctess_1, lctess_nn, k2id, tid):
    toff        = int(lck2[0,0])
    fig     = plt.figure()
    plt.rcParams["font.family"] = "Arial"   # 使用するフォント
    plt.rcParams["font.size"] = 10  
    ax1     = fig.add_subplot(2,1,1)
    ax1.scatter(lck2[0]-toff,lck2[1],s=0.7,c="black")
    ax1.scatter(lck2_1[0]-toff,lck2_1[1]+1,s=0.5,c="dimgrey")
    ax1.scatter(lck2_1[0]-toff,lck2_nn[1]+1,s=0.3,c="orangered")

    k2m,k2u,k2l = mes_amplitude(lck2_nn[1])
    ax1.axhline(k2m+1, c='black',ls='--',lw=0.5)
    ax1.axhline(k2u+1, c='black',ls=':',lw=0.5)
    ax1.axhline(k2l+1, c='black',ls=':',lw=0.5)

    ax2     = fig.add_subplot(2,1,2)
    ax2.scatter(lctess[0]-toff,lctess[1],s=0.7,c="black")
    ax2.scatter(lctess_1[0]-toff,lctess_1[1]+1,s=0.5,c="dimgrey")
    ax2.scatter(lctess_1[0]-toff,lctess_nn[1]+1,s=0.3,c="orangered")

    tsm,tsu,tsl = mes_amplitude(lctess_nn[1])
    ax2.axhline(tsm+1, c='black',ls='--',lw=0.5)
    ax2.axhline(tsu+1, c='black',ls=':',lw=0.5)
    ax2.axhline(tsl+1, c='black',ls=':',lw=0.5)
    fig.suptitle("EPIC "+ k2id+ " / "+ tid)
    fig.supxlabel('time - '+str(toff)+" [d]")
    fig.supylabel('relative flux')
    fig.tight_layout()

    plt.savefig("figure/"+k2id+".png", dpi=200)

def lc_clean(lc, sepscale=10):
    sp_lc   = lt.split_discon(lc, scale=sepscale)
    dl      = []
    for lc_parts in sp_lc:
        det_lc  = lt.detrend_lc(lc_parts, npoint=5)
        cln_lc  = lt.remove_outlier(det_lc, nsigma=5)
        dl.append(cln_lc)
    dl_st   = np.hstack(dl)
    return dl_st

def mes_amplitude(flux):
    fsort   = np.sort(flux)
    flen    = len(fsort)
    fu      = fsort[int(flen*0.95)]
    fl      = fsort[int(flen*0.05)]
    return np.median(flux), fl, fu


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
                np.savetxt("lightcurves/"+k2id+"_k2.dat",lck2.T)
                lctess  = gl.tesslc_byepic(k2id)
                np.savetxt("lightcurves/"+k2id+"_tess.dat",lctess.T)
            
            lck2_1      = lc_clean(lck2, 100)
            lck2_nn     = ft.rm_whitenoise(lck2_1,3)
            wnk2        = lck2_1[1] - lck2_nn[1]
            _,b,c       = mes_amplitude(wnk2)
            erk2        = np.abs(b - c)/2.
            _,b,c       = mes_amplitude(lck2_nn[1])
            ampk2       = np.abs(b - c)/2.

            lctess_1    = lc_clean(lctess, 1e8)
            lctess_nn   = ft.rm_whitenoise(lctess_1,3)
            wntess      = lctess_1[1] - lctess_nn[1]
            _,b,c       = mes_amplitude(wntess)
            ertess      = np.abs(b - c)/2.
            _,b,c       = mes_amplitude(lctess_nn[1])
            amptess     = np.abs(b - c)/2.

            #output      = np.array([k2id, ampk2, erk2, amptess, ertess], dtype='f8')
            output      = np.array([k2id, ampk2, erk2, amptess, ertess], dtype='unicode')
            
            out_array.append(output)

            #ft.plot_freq(lck2_1)
            #ft.plot_freq(lctess_1)
            #plt.errorbar(ampk2, amptess, xerr=erk2, yerr=ertess, fmt='.')

            plotfunc(lck2, lck2_1, lck2_nn, lctess, lctess_1, lctess_nn,\
                    k2id, tid)

            
            i+=1
            if i==10:
                outfilename = "result/" + fname.split(".")[0] + "_out.dat"
                np.savetxt(outfilename, np.array(out_array, dtype='f8'), fmt='%s')
                #plt.plot(np.linspace(0,0.05,10), np.linspace(0,0.05,10), lw=1, c='black', ls='--')
                #plt.xlim((0,0.02))
                #plt.ylim((0,0.02))
                #plt.show()
                exit()

