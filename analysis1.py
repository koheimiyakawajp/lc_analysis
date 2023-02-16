#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import sys
from copy import copy


def plot_vj_MK(prpdata, ax=np.nan):

    k2id   = np.array(prpdata[1:,0], dtype='f8')
    j   = np.array(prpdata[1:,1], dtype='f8')
    jer = np.array(prpdata[1:,2], dtype='f8')
    k   = np.array(prpdata[1:,5], dtype='f8')
    ker = np.array(prpdata[1:,6], dtype='f8')
    v   = np.array(prpdata[1:,7], dtype='f8')
    rw  = np.array(prpdata[1:,11], dtype='f8')
    ga  = np.array(prpdata[1:,12], dtype='f8')
    d   = np.array(prpdata[1:,13], dtype='f8')
    plx = prpdata[1:,8]
    per = prpdata[1:,9]
    
    k2id   = k2id[(plx!=('--'))]
    j   = j[(plx != '--')]
    jer = jer[(plx != '--')]
    k   = k[(plx != '--')]
    ker = ker[(plx != '--')]
    v   = v[(plx != '--')]
    rw  = rw[(plx != '--')]
    ga  = ga[(plx != '--')]
    d   = d[(plx != '--')]
    vj  = v-j
    per = per[(plx!='--')]
    plx = plx[(plx!='--')]
    plx = np.array(plx, dtype='f8')
    per = np.array(per, dtype='f8')
    MK  = k + 5 + 5*np.log10(plx*1e-3)
    MKp = k+ker + 5 + 5*np.log10((plx-per)*1e-3)
    MKm = k-ker + 5 + 5*np.log10((plx+per)*1e-3)
    MKe = np.abs((MKp - MKm)/2.)

    mksar   = np.array((np.sort(vj), MK[(np.argsort(vj))], jer[(np.argsort(vj))],\
         MKe[(np.argsort(vj))],np.arange(len(vj))[np.argsort(vj)]))
    mkmed   = medfilt(mksar[1],kernel_size=11)
    resid   = mksar[1] - mkmed

    con1    = ((rw< 1.4)&((ga< 20)|(d< 5)))
    con1    = con1[np.argsort(vj)]
    con2    = ((rw>=1.4)|((ga>=20)&(d>=5)))
    con2    = con2[np.argsort(vj)]

    if ax is not np.nan:
        mkok    = copy(mksar[:,((resid<2)|con1)])
        mkng    = copy(mksar[:,((resid>=2)|con2)])
        ax.errorbar(mkok[0], mkok[1], xerr=mkok[2], yerr=mkok[3], c='orange', fmt='o', capsize=3, ecolor='gray', markeredgecolor='gray', markersize=5)
        ax.errorbar(mkng[0], mkng[1], xerr=mkng[2], yerr=mkng[3], c='black',  fmt='o', capsize=3, ecolor='gray', markeredgecolor='gray', markersize=5)
        ax.set_xlabel("$V~-~J$")
        ax.set_ylabel("M$_K$")
    
    flg     = np.where((resid >=2)|(con2), 1, 0)
    flgsort = flg[np.argsort(mksar[-1])]
    return  np.array(np.array(k2id[(flgsort==1)],dtype='i8'), dtype='unicode')

def plot_teff_mass(ax, tf, ms, tf_er, ms_er):
    ax.scatter(tf,ms,c="orange",s=30,ec='gray',zorder=3,label="RUWE<1.4")
    ax.errorbar(tf,ms,xerr=tf_er,yerr=ms_er,fmt='.',c='gray',capsize=3,zorder=1)#,c=rw,s=30,cmap=plt.cm.jet,ec='gray',vmin=1,vmax=1.5)
    ax.set_xlabel("Effective Temperature [K]")
    ax.set_ylabel("Stellar Mass [M$_{\odot}$]")

def cal_err_wari(a,b,ae,be):
    er  = ((ae/b)**2+(a*be/b**2)**2)**0.5
    return er

if __name__=='__main__':

    print("hellow world")

    if len(sys.argv) == 3:
        ampfile     = sys.argv[1]
        prpfile     = sys.argv[2]

        ampdata     = np.loadtxt(ampfile, dtype='f8', comments='#')
        prpdata     = np.loadtxt(prpfile, dtype='unicode', comments='#', delimiter=',') 
        prptit      = prpdata[0]
        #print(prptit)
        #exit()
        fig     = plt.figure(figsize=(9,6.5))
        ax1     = fig.add_subplot(2,2,1)
        rmids   = plot_vj_MK(prpdata,ax1)
        for rmid in rmids:
            prpdata = prpdata[(prpdata[:,0] != rmid)]
        prpval      = prpdata[1:]

        k2id   = np.array(prpval[:,0], dtype='f8')
        tf  = np.array(prpval[:,16], dtype='f8')
        ms  = np.array(prpval[:,14], dtype='f8')
        tf_er  = np.array(prpval[:,17], dtype='f8')
        ms_er  = np.array(prpval[:,15], dtype='f8')
        rw  = np.array(prpval[:,11], dtype='f8')
        ga  = np.array(prpval[:,12], dtype='f8')
        d   = np.array(prpval[:,13], dtype='f8')
        #ax1.scatter(tf[(rw<1.4)],ms[(rw<1.4)],c="orange",s=30,cmap=plt.cm.jet,lw=1,ec='gray',vmin=1,vmax=1.5,zorder=2)
        ax2     = fig.add_subplot(2,2,2)
        plot_teff_mass(ax2, tf, ms, tf_er, ms_er)
        #plt.tight_layout()
        #plt.show()
        #exit()

        #print(ampdata,prpdata)
        res     = []
        for dline in ampdata:
            epic    = dline[0]
            con     = (np.array(prpval[:,0], dtype='f8') == epic)
            if np.any(con) :
                prop    = prpval[(np.array(prpval[:,0], dtype='f8') == epic)][0]

                teff    = float(prop[(prptit=="teff")][0])
                teff_er = float(prop[(prptit=="teff_er")][0])
                mass    = float(prop[(prptit=="mass")][0])
                mass_er = float(prop[(prptit=="mass_er")][0])
                ruwe    = float(prop[(prptit=="ruwe")][0])
                res.append(np.hstack((dline, teff, teff_er, mass, mass_er, ruwe)))

        res     = np.array(res,dtype='f8')
        print(len(res[:,0]))
        ax3     = fig.add_subplot(2,2,3)
        ax3.errorbar(res[:,1], res[:,3], xerr=res[:,2], yerr=res[:,4],  zorder=1,\
            fmt='.', capsize=3, c='lightgray', elinewidth=0.5)
        ax3.plot(np.linspace(1e-4,1e-1), np.linspace(1e-4,1e-1), c='black', ls=':', lw=1)
        mp2     = ax3.scatter(res[:,1], res[:,3],  c=res[:,7], s=30, zorder=3, ec='gray',cmap=plt.cm.plasma)
        ax3.set_xlabel("$h_{Kp}$ : Semi-Amplitude in Kp")
        ax3.set_ylabel("$h_T$ : Semi-Amplitude in T")
        ax3.set_xlim((2e-4, 1e-1));ax3.set_ylim((2e-4, 1e-1))
        ax3.set_xscale("log");ax3.set_yscale("log")
        cb1 = fig.colorbar(mp2, ax=ax3)
        cb1.set_label("Effective Temperature [K]")



        ax4     = fig.add_subplot(2,2,4) 
        yer     = cal_err_wari(res[:,3], res[:,1], res[:,4], res[:,2])
        #print(yer)
        #exit()
        ax4.errorbar(res[:,7], res[:,3]/res[:,1], xerr=res[:,8], yerr=yer,  zorder=1, \
            fmt='.', capsize=3, c='lightgray', elinewidth=0.5)
        mp      = ax4.scatter(res[:,7], res[:,3]/res[:,1], c=res[:,-1], s=30, ec='gray' ,zorder=3)
        ax4.set_yscale("log")
        ax4.set_ylabel("$h_{T}/h_{Kp}$")
        ax4.set_xlabel("Effective Temperature [K]")
        cb2     = fig.colorbar(mp, ax=ax4)
        cb2.set_label("RUWE")

        #ax4     = fig.add_subplot(2,2,4) 
        #ax4.scatter(res[:,9], res[:,3]/res[:,1], s=15)
        #ax4.set_yscale("log")
        #ax4.set_ylabel("$h_{T}/h_{Kp}$")
        #ax4.set_xlabel("Stellar Mass [M$_\odot$]")
        plt.tight_layout()
        plt.show()
        exit()
        #plt.scatter(res[:,17],res[:,-1], s=15)
        #plt.scatter(res[:,17], res[:,3]/res[:,1], s=15)
        #plt.scatter(res[:,17], res[:,5]/res[:,1], s=15)
        #plt.scatter(res[:,23], res[:,5]/res[:,1], s=15)
        #plt.scatter(res[:,21], res[:,3]/res[:,1], s=10)
        #plt.scatter(res[:,21], res[:,5]/res[:,1], s=10)
        #plt.scatter(res[:,1], res[:,3],s=10)
        #plt.scatter(res[:,1], res[:,5],s=10)
        #plt.plot(np.linspace(2e-4,2e-1,100),np.linspace(2e-4,2e-1,100),lw=1,ls='--',c='black')
        #plt.xlim((2e-4,2e-1));plt.ylim((2e-4,2e-1))
        #plt.xlim((2e-4,2e-1));plt.ylim((2e-4,2e-1))
        plt.yscale("log");plt.xscale("log")
        plt.show()
        #exit()
        

