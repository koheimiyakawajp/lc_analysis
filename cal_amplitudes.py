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

def ylimits(ydata):
    med     = np.median(ydata)
#    mad     = np.median(np.abs(np.median(ydata) - ydata))
    ysort   = np.sort(ydata)
    dlen    = len(ydata)
    sig1    = ysort[int(dlen*0.1585)] - med
    sig2    = ysort[-int(dlen*0.1585)] - med

    return med+4*sig1, med+4*sig2

def plotfunc3(lck2 ,lck2_1, lck2_nn,  lctess, lctess_1, lctess_nn, lctqlp, lctqlp_1, lctqlp_nn, k2id, tid):
    toff        = int(lck2[0,0])
    fig     = plt.figure(figsize=(5,6))
    #plt.rcParams["font.family"] = "cmss10"   # 使用するフォント
    plt.rcParams["font.size"] = 10  
    ax1     = fig.add_subplot(3,1,1)
    ax1.scatter(lck2[0]-toff,lck2[1],s=0.7,c="black")
    ax1.scatter(lck2_1[0]-toff,lck2_1[1]+1,s=0.5,c="dimgrey")
    ax1.scatter(lck2_1[0]-toff,lck2_nn[1]+1,s=0.3,c="orangered")

    k2m,k2u,k2l = mes_amplitude(lck2_nn[1])
    ax1.axhline(k2m+1, c='black',ls='--',lw=0.5)
    ax1.axhline(k2u+1, c='black',ls=':',lw=0.5)
    ax1.axhline(k2l+1, c='black',ls=':',lw=0.5)
    ax1.set_ylim((ylimits(lck2[1])))

    ax2     = fig.add_subplot(3,1,2)
    ax2.scatter(lctess[0]-toff,lctess[1],s=0.7,c="black")
    ax2.scatter(lctess_1[0]-toff,lctess_1[1]+1,s=0.5,c="dimgrey")
    ax2.scatter(lctess_1[0]-toff,lctess_nn[1]+1,s=0.3,c="orangered")

    tsm,tsu,tsl = mes_amplitude(lctess_nn[1])
    ax2.axhline(tsm+1, c='black',ls='--',lw=0.5)
    ax2.axhline(tsu+1, c='black',ls=':',lw=0.5)
    ax2.axhline(tsl+1, c='black',ls=':',lw=0.5)
    ax2.set_ylim((ylimits(lctess[1])))

    ax3     = fig.add_subplot(3,1,3)
    ax3.scatter(lctqlp[0]-toff,  lctqlp[1],s=0.7,c="black")
    ax3.scatter(lctqlp_1[0]-toff,lctqlp_1[1]+1,s=0.5,c="dimgrey")
    ax3.scatter(lctqlp_1[0]-toff,lctqlp_nn[1]+1,s=0.3,c="orangered")

    tsm,tsu,tsl = mes_amplitude(lctqlp_nn[1])
    ax3.axhline(tsm+1, c='black',ls='--',lw=0.5)
    ax3.axhline(tsu+1, c='black',ls=':',lw=0.5)
    ax3.axhline(tsl+1, c='black',ls=':',lw=0.5)
    ax3.set_ylim((ylimits(lctqlp[1])))


    fig.suptitle("EPIC "+ k2id+ " / "+ tid)
    fig.supxlabel('time - '+str(toff)+" [d]");fig.supylabel('relative flux')
    fig.tight_layout()

    plt.savefig("figure/"+k2id+".png", dpi=200)
    plt.clf();plt.close()
    #plt.show()
    #exit()


def plotfunc2(lck2 ,lck2_1, lck2_nn, lctess, lctess_1, lctess_nn, k2id, tid):
    toff        = int(lck2[0,0])
    fig     = plt.figure(figsize=(5,4))
    #plt.rcParams["font.family"] = "cmss10"   # 使用するフォント
    plt.rcParams["font.size"] = 10  
    ax1     = fig.add_subplot(2,1,1)
    ax1.scatter(lck2[0]-toff,lck2[1],s=0.7,c="black")
    ax1.scatter(lck2_1[0]-toff,lck2_1[1]+1,s=0.5,c="dimgrey")
    ax1.scatter(lck2_1[0]-toff,lck2_nn[1]+1,s=0.3,c="orangered")

    k2m,k2u,k2l = mes_amplitude(lck2_nn[1])
    ax1.axhline(k2m+1, c='black',ls='--',lw=0.5)
    ax1.axhline(k2u+1, c='black',ls=':',lw=0.5)
    ax1.axhline(k2l+1, c='black',ls=':',lw=0.5)
    ax1.set_ylim((ylimits(lck2[1])))

    ax2     = fig.add_subplot(2,1,2)
    ax2.scatter(lctess[0]-toff,lctess[1],s=0.7,c="black")
    ax2.scatter(lctess_1[0]-toff,lctess_1[1]+1,s=0.5,c="dimgrey")
    ax2.scatter(lctess_1[0]-toff,lctess_nn[1]+1,s=0.3,c="orangered")

    tsm,tsu,tsl = mes_amplitude(lctess_nn[1])
    ax2.axhline(tsm+1, c='black',ls='--',lw=0.5)
    ax2.axhline(tsu+1, c='black',ls=':',lw=0.5)
    ax2.axhline(tsl+1, c='black',ls=':',lw=0.5)
    ax2.set_ylim((ylimits(lctess[1])))

    fig.suptitle("EPIC "+ k2id+ " / "+ tid)
    fig.supxlabel('time - '+str(toff)+" [d]");fig.supylabel('relative flux')
    fig.tight_layout()

    #plt.show()
    #exit()
    plt.savefig("figure/"+k2id+".png", dpi=200)
    plt.clf();plt.close()

def lc_clean(lc, sepscale=10):
    sp_lc   = lt.split_discon(lc, scale=sepscale)
    dl      = []
    #plt.scatter(lc[0],lc[1])
    #plt.scatter(sp_lc[0][0],sp_lc[0][1])
    #plt.show()
    #exit()
    for lc_parts in sp_lc:
        binsep  = 10 #day
        trange  = lc_parts[0,-1] - lc_parts[0,0]
        npoint  = int(trange/binsep)
        if npoint>2:
            det_lc  = lt.detrend_lc(lc_parts, npoint=npoint)
        else:
            det_lc  = lc_parts
            det_lc[1]  -= np.median(det_lc[1])
        #print("***** ", np.median(det_lc[1]))
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

def roop_mes(data_nn, pbest):
    dlen    = data_nn[0,-1] - data_nn[0,0]
    roopn   = int(dlen/pbest) + 1
    t_beg   = data_nn[0,0]

    amp_ar  = []
    for i in range(roopn):
        t_a     = t_beg + i*pbest
        t_b     = t_beg + (i+1)*pbest
        #plt.axvline(t_b)
        con     = ((t_a<=data_nn[0])&(data_nn[0]<t_b))
        d_i     = data_nn[:,con]
        if np.any(con):
            #plt.scatter(d_i[0], d_i[1], s=2)
            if (d_i[0,-1] - d_i[0,0]) > 0.9*pbest:
                _,b,c   = mes_amplitude(d_i[1])
                amp     = np.abs(b-c)/2.
                amp_ar.append(amp)
    if len(amp_ar) == 0:
        return np.nan, np.nan
    elif len(amp_ar) == 1:
        return amp, np.nan
    else:
        return np.mean(amp_ar), np.std(amp_ar)

def mes_wrap(data, pbest, wsigma=3):
    data_nn     = ft.rm_whitenoise(data,wsigma)

    #plt.scatter(data[0], data[1], s=1)
    #print(roop_mes(data_nn, pbest))
    amp,er  = roop_mes(data_nn, pbest)
    #plt.scatter(data_nn[0], data_nn[1], s=1)
    #plt.show()
    #data2       = data[1] - data_nn[1]
    #_,b,c       = mes_amplitude(data2)
    #er          = np.abs(b - c)/2.
    #_,b,c       = mes_amplitude(data_nn[1])
    #amp         = np.abs(b - c)/2.

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
    #trange      = data[0,-1] - data[0,0]
    #pmax        = trange//2
    pmax        = 35
    pmin        = 0.1
    ngrid       = pmax//0.005
    _,_,p,pow   = pr.lomb_scargle(data,N=int(ngrid),pmin=pmin,pmax=pmax)
    pgm         = np.array((p,pow))
    print("calculating sigmin val.")
    sigmin      = pr.sigmin_bootstrap(data,N=int(ngrid),pmin=pmin,pmax=pmax,nboot=100, seed=300)
    #sigmin  = 1e-3
    peaks   = argrelextrema(pow, np.greater)

    pgm_peak_ok = copy(pgm[:,peaks])
    #print(pgm_peak_ok)
    pgm_peak_ok = pgm_peak_ok[:,(pgm_peak_ok[1]>sigmin)]#abovethres
    if len(pgm_peak_ok[0]) == 0:
        return np.nan
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
    #plt.rcParams["font.family"] = "cmss10"   # 使用するフォント
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
    out_array   = [["#ID", "amp_k2", "er_k2", "amp_tess", "er_tess", "amp_qlp", "er_qlp"]]
    for k2id in epiclist:
        tid     = gl.EPIC_to_TIC(k2id)
        if tid != -1:
            print("EPIC "+k2id, tid)
            fkey    = "lightcurves/"+k2id
            flg     = 0
            if os.path.isfile(fkey+"_k2.dat"): #----------------------
                print("loading k2 lightcurve.")
                lck2    = np.loadtxt(fkey+"_k2.dat", dtype='f8').T
                if len(lck2)!=0:
                    print("processing K2 data.")
                    lck2_1      = lc_clean(lck2, 100)
                    #ft.plot_freq(lck2_1)
                    fileperi    = "period/"+k2id+"_k2.dat"
                    if os.path.isfile(fileperi):
                        print("period file for K2 data already exists.")
                        pres_k2     = np.loadtxt(fileperi, dtype='f8')
                    else:
                        print("running period analysis for K2 data.")
                        pres_k2     = period_analysis(lck2_1,k2id + " K2")
                        if pres_k2 is not np.nan:
                            np.savetxt(fileperi, pres_k2)
                    if pres_k2.ndim == 2:
                        pbest   = pres_k2[0,0]
                    elif pres_k2.ndim == 1:
                        pbest   = pres_k2[0]
                    print("measuring amplitude for K2 data.")
                    lck2_nn,ampk2,erk2  = mes_wrap(lck2_1, pbest, wsigma=3)
                    flg     = 1
                
            if os.path.isfile(fkey+"_tess.dat"): #--------------------
                print("loading tess lightcurve.")
                lctess  = np.loadtxt(fkey+"_tess.dat", dtype='f8').T
                if len(lctess)!=0:

                    print("processing TESS data.")
                    lctess_1    = lc_clean(lctess, 1e8)
                    #print("running period analysis for TESS data.")
                    #pres_tess   = period_analysis(lctess_1,k2id + " TESS")
                    print("measuring amplitude for TESS data.")
                    lctess_nn,amptess,ertess  = mes_wrap(lctess_1, pbest, wsigma=3)
                    print(amptess, ertess)

                    flg     += 2
            else:
                lctess_nn,amptess,ertess  = np.nan,np.nan,np.nan

            if os.path.isfile(fkey+"_tess_qlp.dat"): #-----------------
                print("loading tess_qlp lightcurve.")
                lctqlp  = np.loadtxt(fkey+"_tess_qlp.dat", dtype='f8').T
                if len(lctqlp)!=0:

                    print("processing TESS QLP data.")
                    lctqlp_1    = lc_clean(lctqlp, 100)
                    fileperi    = "period/"+k2id+"_tess_qlp.dat"
                    if os.path.isfile(fileperi):
                        print("period file for TESS QLP data already exists.")
                        pres_tqlp   = np.loadtxt(fileperi, dtype='f8')
                    else:
                        print("running period analysis for TESS QLP data.")
                        pres_tqlp   = period_analysis(lctqlp_1,k2id + " TESS_QLP")
                        if pres_tqlp is not np.nan:
                            np.savetxt(fileperi, pres_tqlp)

                    print("measuring amplitude for TESS QLP data.")
                    lctqlp_nn,amptqlp,ertqlp  = mes_wrap(lctqlp_1, pbest, wsigma=3)

                    flg     += 3
            else:
                lctqlp_nn,amptqlp,ertqlp  = np.nan,np.nan,np.nan

            if flg>=3:
                output      = np.array([k2id, ampk2, erk2, amptess, ertess, amptqlp, ertqlp], dtype='unicode')
                out_array.append(output)

                #ft.plot_freq(lck2_1)
                #ft.plot_freq(lctess_1)
                #plt.errorbar(ampk2, amptess, xerr=erk2, yerr=ertess, fmt='.')
            if flg==6:
                plotfunc3(lck2, lck2_1, lck2_nn, lctess, lctess_1, lctess_nn,\
                    lctqlp, lctqlp_1, lctqlp_nn, k2id, tid)
            elif flg==3:
                plotfunc2(lck2, lck2_1, lck2_nn, lctess, lctess_1, lctess_nn,\
                        k2id, tid)
            elif flg==4:
                plotfunc2(lck2, lck2_1, lck2_nn, lctqlp, lctqlp_1, lctqlp_nn,\
                        k2id, tid+" (QLP)")

                
                #i+=1
                #if i==20:
                #    outfilename = "result/" + fname.split(".")[0] + "_out.dat"
                #    np.savetxt(outfilename, np.array(out_array, dtype='f8'), fmt='%s')
    outfilename = "result/" + fname.split(".")[0] + "_out.dat"
    np.savetxt(outfilename, out_array, fmt='%s')
            #    #plt.plot(np.linspace(0,0.05,10), np.linspace(0,0.05,10), lw=1, c='black', ls='--')
            #    #plt.xlim((0,0.02))
            #    #plt.ylim((0,0.02))
            #    #plt.show()
            #    exit()

