import numpy as np
from matplotlib import pyplot as plt

def fft(data):
    time    = data[0]
    flux    = data[1]
    samp_t  = np.median(time[1:] - time[:-1])
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

def det_maxpeak(p,pow):
    p_best  = p[(pow==max(pow))]
    i_p     = int(np.where(p==p_best)[0])
    i_min   = i_p
    i_max   = i_p
    for i in range(i_p):
        if pow[i_max-1] >= pow[i_max]:
            break
        else:
            i_max -= 1
    for i in range(len(p) - i_p-1):
        if pow[i_min+1] >= pow[i_min]:
            break
        else:
            i_min += 1

    return float(p_best), p[i_min], p[i_max]


def fftmaxpeak_filter(data):
    fft_data    = fft(data)

    N   = len(data[0])
    abs_f       = np.abs(fft_data[1])
    abs_f       = abs_f/N*2
    abs_f[0]    = abs_f[0]/2
    
    fft_time    = fft_data[0]
    fft_flux    = fft_data[1]
    plt.plot(np.abs(fft_time[:int(N/2)-1]), abs_f[:int(N/2)-1])
    plt.xscale("log")
    plt.show()

    fbest,fmax,fmin     = det_maxpeak(np.abs(fft_time[:int(N/2)-1]), abs_f[:int(N/2)-1])    

    fft_flux[(fmin < fft_time)&(fft_time < fmax)] = np.median(fft_flux)
    fft_flux[(fft_time[-1] - fmax < fft_time)&(fft_time < fft_time[-1] - fmin)] =np.median(fft_flux)
    new_data    = ifft(fft_time, fft_flux)

    return new_data, fbest, fmax, fmin
    


def peak_filter(data, freq_array):
    fft_data    = fft(data)
    fft_time    = fft_data[0]
    fft_flux    = fft_data[1]
    for fp in freq_array:
        fu  = fp*1.1
        fl  = fp*0.9
        fft_flux[((fl < fft_time)&\
                  (fft_time < fu))] = 0
        fft_flux[((fft_time[-1] - fu < fft_time)&\
                  (fft_time < fft_time[-1] - fl))] = 0
    new_data    = ifft(fft_time, fft_flux)
    new_data[0] = new_data[0] + data[0,0]

    return new_data

def rm_whitenoise(data, nsigma):
    fft_data    = fft(data)
    fft_time    = fft_data[0]
    fft_flux    = fft_data[1]

    N           = len(fft_flux)

    abs_fft_flux    = np.abs(fft_flux)
    abs_fft_flux    = abs_fft_flux / N * 2 # 交流成分
    abs_fft_flux[0] = abs_fft_flux[0] / 2     # 直流成分

    median      = np.median(abs_fft_flux)
    wh_level    = np.median(np.abs(abs_fft_flux - median))*1.48*nsigma

    fft_flux[(abs_fft_flux < median+wh_level)]  = 0
    new_data    = ifft(fft_time, fft_flux)
    #new_data[0] = new_data[0] + data[0,0]
    new_data[0] = data[0]

    return new_data, wh_level


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

