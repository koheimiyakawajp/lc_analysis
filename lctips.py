import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import copy

def split_discon(data):
    time    = data[0]

    dif_t   = time[1:] - time[:-1]
    sep     = np.median(dif_t)

    ids     = np.where(dif_t > sep*10)[0]
    ids     += 1

    i0      = 0
    split_array     = []
    for i in ids:
        #plt.plot(data[0,i0:i], data[1,i0:i])
        split_array.append(data[:,i0:i])
        i0  = i
    #plt.show()
    split_array.append(data[:,i0:])
    return split_array

def exten3_lc(data):
    x1  = data[0]*(-1.) + 2*data[0,0]
    x2  = data[0]*(-1.) + 2*data[0,-1]

    data1   = np.vstack((x1,data[1:]))
    data2   = np.vstack((x2,data[1:]))

    data1   = data1[:,::-1]
    data2   = data2[:,::-1]

    d_trpl  = np.hstack((data1,data,data2))

    return d_trpl

def med_bin(array, k):
    dlen    = len(array[0])
    k       = int(k)
    N_k     = dlen//k

    #res_array   = np.array(((1.,1.)), dtype='f8')
    res_array   = []
    for i in range(k):
        tmp_ar  = array[:,i*N_k:(i+1)*N_k]
        tmp_med = np.median(tmp_ar, axis=1)
        res_array.append(tmp_med)

    res_array   = np.array(res_array)
    return res_array.T

def remove_flare(data, nsigma=4):
    med     = np.median(data[1])
    mad     = np.median(np.abs(data[1] - med))
    std     = 1.48*mad
    thres_p = med + std*nsigma
    thres_n = med - std*nsigma

    data[1,((data[1]<thres_n)|(thres_p<data[1]))] = med
    return data



def detrend_lc(data, npoint=10):
    data    = remove_flare(data)
    data[1] = data[1] - np.median(data[1])
    d_tr    = exten3_lc(data)
    medd    = med_bin(d_tr, npoint)

    fn      = interpolate.interp1d(medd[0], medd[1], kind='cubic')
    trend   = fn(data[0])
    trend   = trend - np.median(trend)
    #plt.plot(data[0], data[1])
    #plt.plot(data[0], trend -1)

    datad   = np.copy(data)
    datad[1]= data[1] - trend
    #plt.plot(data[0], data_dt-data[1])
    #plt.plot(data[0], data_dt)
    #plt.show()

    return datad





