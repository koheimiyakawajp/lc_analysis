import numpy as np


def fft(data):
    time    = data[0]
    flux    = data[1]
    samp_t  = time[2] - time[1]
    N = len(flux)

    fft_flux        = np.fft.fft(flux)
    freq    = np.linspace(0, 1.0/samp_t, N)

    fft_data    = np.array((freq, fft_flux))
    return fft_data


def lowpass(fft_time, fft_flux, cut_off):

    N   = len(fft_flux)
    fft_flux[(fft_time > cut_off)]  = 0

    return fft_time,fft_flux

def ifft(fft_time, fft_flux):
    N       = len(fft_flux)
    flux    = np.fft.ifft(fft_flux)

    flux    = flux.real
    freq    = fft_time.real[-1]
    print(freq)
    time    = np.arange(0, 1.0/freq*N, 1.0/freq)

    print(time)
    print(flux)
    return np.array((time, flux), dtype='f8')

def lowpass_filter(data, cutoff):
    fft_data    = fft(data)
    fft_time, fft_flux  = lowpass(fft_data[0], fft_data[1], cutoff)
    new_data    = ifft(fft_time, fft_flux)

    return new_data



