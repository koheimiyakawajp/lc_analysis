import numpy as np
from scipy import signal as sg
from matplotlib import pyplot as plt
from transitleastsquares import transitleastsquares as tls
from transitleastsquares import transit_mask
import copy

def fap(pgram, prob):
    n   = len(pgram)
    med = np.median(pgram)
    std = 1.48*np.median(np.abs(pgram-med))
    m   = len(pgram[(med+std*3 < pgram)])
    val = 1-np.power(1-np.power(1-prob,1./m),2./(n-3.))

    return val

def find_peak(freq, pgram, thres=0.01):
    peaks       = pgram[(thres<pgram)]
    freq_p      = freq[(thres<pgram)]

    return freq_p, peaks

def lomb_scargle(data, N=1000, pmin=0.1, pmax=10., prob=0.01):
    time    = data[0]
    flux    = data[1]

    timein  = time - time[0]
    pmin    = pmin*2*np.pi
    pmax    = pmax*2*np.pi
    freq    = np.linspace(pmin, pmax, N)
    pgram   = sg.lombscargle(timein, flux, freq)

    thres   = fap(pgram, prob)

    f,p     = find_peak(freq, pgram, thres)

    return f/2/np.pi,p,freq/2/np.pi,pgram

def tls_periodogram(data, rad_star=1., mas_star=1.):
    data[1]     = data[1] - np.median(data[1]) + 1.
    model       = tls(data[0], data[1])
    pgram       = model.power(R_star=rad_star, M_star=mas_star,\
                              duration_grid_step=2., oversampling_factor=1)

    return np.array((pgram.periods, pgram.power)), pgram.SDE

def mask_transit(data, period, duration, T0):
    med         = np.median(data[1])
    intransit   = transit_mask(data[0], period, duration, T0)
    data_ot     = copy.copy(data)
    data_ot[1:,intransit]    = med

    return data_ot


