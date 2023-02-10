#!/usr/bin/env python3

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from copy import copy
from scipy.signal import argrelextrema

import bin.getlc as gl
import bin.lctips as lt
import bin.fft as ft
import bin.period as pr

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

def mes_wrap(data, wsigma=3):
    data_nn     = ft.rm_whitenoise(data,wsigma)
    data2       = data[1] - data_nn[1]
    _,b,c       = mes_amplitude(data2)
    er          = np.abs(b - c)/2.
    _,b,c       = mes_amplitude(data_nn[1])
    amp         = np.abs(b - c)/2.

    return data_nn,amp, er

def peri_error_single(p,pow,p_best):
    i_p     = int(np.where(p==p_best)[0])
    i_min   = i_p
    i_max   = i_p
    for i in range(i_p):
        if (pow[i_max-1] >= pow[i_max]):
            break
        else:
            i_max -= 1
    for i in range(len(p) - i_p-1):
        if (pow[i_min+1] >= pow[i_min]):
            break
        else:
            i_min += 1

    return p[i_min], p[i_max]

def peri_error_thres(p,pow,p_best,thres):
    i_p     = int(np.where(p==p_best)[0])
    i_min   = i_p
    i_max   = i_p
    for i in range(i_p):
        if (pow[i_max-1] < thres):
            break
        else:
            i_max -= 1
    for i in range(len(p) - i_p-1):
        if (pow[i_min+1] < thres):
            break
        else:
            i_min += 1

    return p[i_min]-p_best, p[i_max]-p_best

def remove_harmonics(presult):
    inrange = []
    i       = 0
    presult_filtered    = []
    for pr in presult:
        #print(pr)
        if i==0:
            peri,_,_,_,lim1,lim2  = pr
            inrange.append([lim1,lim2])
            presult_filtered.append(pr)
        else:
            peri,_,er1,er2,lim1,lim2  = pr
            flg     = 0
            for rg in inrange:
                con21    = ((rg[0]/2.<peri+er2)&(peri+er1<rg[1]/2.))
                con22    = ((rg[0]*2.<peri+er2)&(peri+er1<rg[1]*2.))
                con31    = ((rg[0]/3.<peri+er2)&(peri+er1<rg[1]/3.))
                con32    = ((rg[0]*3.<peri+er2)&(peri+er1<rg[1]*3.))
                con41    = ((rg[0]/4.<peri+er2)&(peri+er1<rg[1]/4.))
                con42    = ((rg[0]*4.<peri+er2)&(peri+er1<rg[1]*4.))
                if con21|con22|con31|con32|con41|con42:
                    flg     = 1
                    break
            if flg == 0:
                presult_filtered.append(pr)
                inrange.append([lim1,lim2])

        i+=1
    return np.array(presult_filtered)


def period_analysis(data, title='--'):
    _,_,p,pow   = pr.lomb_scargle(data,N=int(1e4),pmin=0.1,pmax=35)
    pgm         = np.array((p,pow))
    print("calculating sigmin val.")
    sigmin      = pr.sigmin_bootstrap(data,N=int(1e4),pmin=0.1,pmax=35,nboot=100, seed=300)
    #sigmin  = 1e-4
    peaks   = argrelextrema(pow, np.greater)

    pgm_peak_ok = copy(pgm[:,peaks])
    pgm_peak_ok = pgm_peak_ok[:,(pgm_peak_ok[1]>sigmin)]#abovethres
    p1          = pgm[0,(pgm[1]==np.max(pgm[1]))]#best period
    pgm_peak20  = pgm_peak_ok[:,((pgm_peak_ok[0]>=p1*0.80)&(pgm_peak_ok[0]<=p1*1.2))]#pm20% 
    pgm_peakan  = pgm_peak_ok[:,((pgm_peak_ok[0]<p1*0.80)|(pgm_peak_ok[0]>p1*1.2))]#outer than20% 

    pow_sort    = np.hstack((np.sort(pgm_peak20[1])[::-1],np.sort(pgm_peakan[1])[::-1]))
    presult     = []
    for ps in pow_sort:
        if np.any(pgm_peak_ok[1] == ps):
            p_best      = float(pgm_peak_ok[0,(pgm_peak_ok[1]==ps)])
            per1,per2 = peri_error_thres(pgm[0],pgm[1],p_best,ps/2.)
            lim1,lim2 = peri_error_single(pgm[0],pgm[1],p_best)
            presult.append([p_best,ps,per1,per2,lim1,lim2])

    presult     = np.array(presult)
    presult2    = remove_harmonics(presult)
    plt.figure(figsize=(5,3))
    plt.rcParams["font.family"] = "Arial"   # 使用するフォント
    plt.rcParams["font.size"] = 10  
    plt.plot(pgm[0],pgm[1],lw=1.,c='black')
    plt.scatter(presult[:,0], presult[:,1],c='royalblue',s=10)
    plt.scatter(presult2[:,0], presult2[:,1],c='orangered',s=10)
    plt.axhline(sigmin, c='blue',ls=':', lw=1)
    plt.xscale('log')
    plt.title(title);plt.xlabel("Period [d]");plt.ylabel("LS Power")
    
    #plt.show()
    tword   = title.split(" ")
    plt.savefig("figure/"+tword[0]+tword[1]+"_pdgram.png", dpi=200)
    plt.clf();plt.close()

    return presult2
        

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
                print("loading lightcurves")
                lck2    = np.loadtxt(fkey+"_k2.dat", dtype='f8').T
                lctess  = np.loadtxt(fkey+"_tess.dat", dtype='f8').T
                
                print("processing K2 data.")
                lck2_1      = lc_clean(lck2, 100)
                #ft.plot_freq(lck2_1)
                print("running period analysis for K2 data.")
                pres_k2     = period_analysis(lck2_1,k2id + " K2")
                print("measuring amplitude for K2 data.")
                lck2_nn,ampk2,erk2  = mes_wrap(lck2_1, wsigma=3)

                print("processing TESS data.")
                lctess_1    = lc_clean(lctess, 1e8)
                print("running period analysis for TESS data.")
                pres_tess   = period_analysis(lctess_1,k2id + " TESS")
                print("measuring amplitude for TESS data.")
                lctess_nn,amptess,ertess  = mes_wrap(lctess_1, wsigma=3)

                output      = np.array([k2id, ampk2, erk2, amptess, ertess], dtype='unicode')
                
                out_array.append(output)

                #ft.plot_freq(lck2_1)
                #ft.plot_freq(lctess_1)
                #plt.errorbar(ampk2, amptess, xerr=erk2, yerr=ertess, fmt='.')

                plotfunc(lck2, lck2_1, lck2_nn, lctess, lctess_1, lctess_nn,\
                        k2id, tid)

                
                i+=1
                if i==20:
                    outfilename = "result/" + fname.split(".")[0] + "_out.dat"
                    np.savetxt(outfilename, np.array(out_array, dtype='f8'), fmt='%s')
            #    #plt.plot(np.linspace(0,0.05,10), np.linspace(0,0.05,10), lw=1, c='black', ls='--')
            #    #plt.xlim((0,0.02))
            #    #plt.ylim((0,0.02))
            #    #plt.show()
            #    exit()

