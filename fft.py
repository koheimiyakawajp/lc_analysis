import numpy as np
from matplotlib import pyplot as plt

def fft(data):
    time    = data[0]
    flux    = data[1]
    samp_t  = time[2] - time[1]
    N = len(flux)

    fft_flux        = np.fft.fft(flux)
    freq    = np.linspace(0, 1.0/samp_t, N)

    fft_data    = np.array((freq, fft_flux))
    return fft_data


def ifft(fft_time, fft_flux):
    N       = len(fft_flux)
    flux    = np.fft.ifft(fft_flux)

    flux    = flux.real
    freq    = fft_time.real[-1]
    time    = np.arange(0, 1.0/freq*N, 1.0/freq)

    return np.array((time, flux), dtype='f8')


def lowpass_filter(data, cutoff):
    fft_data    = fft(data)
    fft_time    = fft_data[0]
    fft_flux    = fft_data[1]
    fft_flux[(cutoff < fft_time)] = 0
    new_data    = ifft(fft_time, fft_flux)
    new_data[0] = new_data[0] + data[0,0]

    return new_data

def highpass_filter(data, cutoff):
    fft_data    = fft(data)
    fft_time    = fft_data[0]
    fft_flux    = fft_data[1]
    fft_flux[(cutoff > fft_time)] = 0
    new_data    = ifft(fft_time, fft_flux)
    new_data[0] = new_data[0] + data[0,0]

    return new_data

def peak_filter(data, freq_array):
    fft_data    = fft(data)
    fft_time    = fft_data[0]
    fft_flux    = fft_data[1]
    for fp in freq_array:
        fu  = fp*1.5
        fl  = fp*0.5
        fft_flux[((fl < fft_time)&\
                  (fft_time < fu))] = 0
        fft_flux[((fft_time[-1] - fu < fft_time)&\
                  (fft_time < fft_time[-1] - fl))] = 0
    new_data    = ifft(fft_time, fft_flux)
    new_data[0] = new_data[0] + data[0,0]

    return new_data

def whitenoise_sigma(data, nsigma):
    fft_data    = fft(data)
    fft_time    = fft_data[0]
    fft_flux    = fft_data[1]

    N           = len(fft_flux)

    abs_fft_flux    = np.abs(fft_flux)
    abs_fft_flux    = abs_fft_flux / N * 2 # 交流成分
    abs_fft_flux[0] = abs_fft_flux[0] / 2     # 直流成分

    median      = np.median(abs_fft_flux)
    wh_level    = np.median(np.abs(abs_fft_flux - median))*1.48*nsigma

    fft_flux[(abs_fft_flux > median+wh_level)]  = 0
    new_data    = ifft(fft_time, fft_flux)
    new_data[0] = new_data[0] + data[0,0]

    return new_data

def plot_freq(data):
    N           = len(data[0])
    fft_data    = fft(data)
    abs_f       = np.abs(fft_data[1])
    abs_f       = abs_f/N*2
    abs_f[0]    = abs_f[0]/2

    plt.plot(fft_data[0, :int(N/2)-1], abs_f[:int(N/2)-1])
    plt.show()

