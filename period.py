import numpy as np
from scipy import signal as sg
from matplotlib import pyplot as plt

def fap(pgram, prob):
    n   = len(pgram)
    med = np.median(pgram)
    std = 1.48*np.median(np.abs(pgram-med))
    m   = len(pgram[(med+std*3 < pgram)])
    val = (1-np.power(1-np.power(1-prob,1./m),2./(n-3.)))
    return val

def find_peak(freq, pgram, prob=0.01):
    fap_val     = fap(pgram, prob)
    peaks       = pgram[(fap_val<pgram)]
    freq_p      = freq[(fap_val<pgram)]
    #p_cen       = []
    #f_cen       = []
    #for i in range(len(peaks)-2):
    #    if (peaks[i]<peaks[i+1])&(peaks[i+2]<peaks[i+1]):
    #        f_cen.append(freq_p[i+1])
    #        p_cen.append(peaks[i+1])

    #p_cen   = np.array(p_cen)
    #f_cen   = np.array(f_cen)
    #return f_cen, p_cen

    return freq_p, peaks

def lomb_scargle(data, N=1000):
    time    = data[0]
    flux    = data[1]

    timein  = time - time[0]
    freq    = np.linspace(1./timein[-1], 20, N)
    pgram   = sg.lombscargle(timein, flux, freq)

    f,p     = find_peak(freq, pgram)

    #plt.plot(freq/2/np.pi, pgram)
    #plt.scatter(f/2/np.pi, p)
    #plt.show()
    #exit()
    return f/2/np.pi,p,freq/2/np.pi,pgram
