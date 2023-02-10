#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import bin.info_from_catalogue as bc
import sys


def Teff_boyajian2012(magV, magJ, FeH):
    X   = magV - magJ
    Y   = FeH

    a0  = 7308
    a1  = -1775
    a2  = 198
    a3  = 71
    a4  = 100
    a5  = 317

    res = a0 + a1*X + a2*X*X + a3*X*Y + a4*Y + a5*Y*Y
    return res, 75

def Teff_mann2015(magV, magJ, FeH):
    X   = magV - magJ
    Y   = FeH

    a   = 2.515
    b   = -1.054
    c   = 0.2965
    d   = -0.04150
    e   = 0.002245
    f   = 0.05262

    res = a + b*X + c*X**2 + d*X**3 + e*X**4 + f*Y
    return res*3500, 42

def Teff_joint(magV, magJ, FeH):
    if magV - magJ < 3:
        teff,er = Teff_boyajian2012(magV, magJ, FeH)
    else:
        teff,er = Teff_mann2015(magV, magJ, FeH)

    return teff, er

dataK07 = np.loadtxt("MK_mass.lst", comments='#', dtype='f8')
def mass_Kraus2007(magK, plx):
    MagK    = magK + 5 + 5*np.log10(plx*1e-3)

    y_K     = interp1d(dataK07[:,2], dataK07[:,-1], kind='linear')
    return  y_K(MagK)

def mass_Mann2019(magK, plx, FeH):
    MagK    = magK + 5 + 5*np.log10(plx*1e-3)

    a0  = -0.647
    a1  = -0.207
    a2  = -6.53e-4
    a3  = 7.13e-3
    a4  = 1.84e-4
    a5  = -1.60e-4
    f   = -0.0035
    zp  = 7.5
    #zp  = 0.0

    x   = MagK - zp
    res = (1+f*FeH)*10**(a0 + a1*x + a2*x**2 + a3*x**3 \
        + a4*x**4 + a5*x**5 )
    
    return res

def mass_joint(magK, plx, FeH):
    mass_M  = mass_Mann2019(magK,plx,FeH)
    if mass_M > 0.7:
        mass_K  = mass_Kraus2007(magK,plx)
        return mass_K
    else:
        return mass_M
def rd(val, ndig=0):
    #print(type(val))
    if (type(val) is float) | (type(val) is np.float64) | (type(val) is np.float32):
        #print(round(val,ndig))
        return round(val,ndig)
    else:
        return val
    
def get_propdict(k2id, FeH=0.0, rad=0.01):

    plx,plx_er,bprp,ruwe,gof_al,d       = bc.get_gaia(k2id,rad)
    
    jmag,jer,hmag,her,kmag,ker  = bc.get_2mass(k2id,rad)
    vmag    = bc.get_tycho(k2id, rad)
    if vmag is np.nan:
        vmag    = bc.get_tic(k2id, rad)
    
    mass        = mass_joint(kmag,plx, FeH)
    mass        = float(mass)
    mass_u      = mass_joint(kmag,plx+plx_er, FeH)
    mass_l      = mass_joint(kmag,plx-plx_er, FeH)
    mass_er1    = (abs(mass_u - mass) + abs(mass - mass_l))/2.
    mass_u      = mass_joint(kmag+ker,plx, FeH)
    mass_l      = mass_joint(kmag-ker,plx, FeH)
    mass_er2    = (abs(mass_u - mass) + abs(mass - mass_l))/2.
    mass_er     = np.sqrt(mass_er1**2 + mass_er2**2)
    mass_er     = float(mass_er)

    teff,teff_er1   = Teff_joint(vmag,jmag,FeH)
    teff            = float(teff)
    teff_u,_        = Teff_joint(vmag,jmag+jer,FeH)
    teff_l,_        = Teff_joint(vmag,jmag-jer,FeH)
    teff_er2        = (abs(teff - teff_u) + abs(teff - teff_l))/2.
    teff_er         = np.sqrt(teff_er1**2 + teff_er2**2)
    teff_er         = float(teff_er)

    return {"k2id": k2id, "jmag": jmag, "jmag_er": jer, "hmag" : hmag, "hmag_er": her,\
        "kmag" : kmag, "kamg_er" : ker, "vmag" : vmag, "plx" : rd(plx,3), "plx_er" : rd(plx_er,3),\
            "bprp" : bprp, "ruwe" : ruwe, "gof_al" : gof_al, "d" : d, "mass" : rd(mass,4), "mass_er" : rd(mass_er,4),\
                "teff" : rd(teff), "teff_er" : rd(teff_er)}


import csv

if __name__=='__main__':
    
    fieldname   =  ["k2id", "jmag", "jmag_er", "hmag", "hmag_er", "kmag", "kamg_er", "vmag",\
        "plx", "plx_er", "bprp", "ruwe", "gof_al", "d", "mass", "mass_er", "teff", "teff_er"]
    if len(sys.argv)==2:

        k2id    = sys.argv[1]
        propdict    = get_propdict(k2id)
        print(propdict)

    elif len(sys.argv)==3:
        fname   = sys.argv[1]
        FeH     = float(sys.argv[2])
        nlist   = np.loadtxt(fname, dtype='unicode',comments='#').T
        
        data    = []
        for k2id in nlist:
            print(k2id)
            dline   = get_propdict(k2id, FeH=FeH, rad=0.1)
            print(dline)
            data.append(dline)

        with open("result/hyades_stellarprop.csv", 'w', encoding='utf-8',newline='')as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = fieldname)
            writer.writeheader()
            writer.writerows(data) 
            
            