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
        split_array.append(data[:,i0:i])
        i0  = i
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

    res_array   = []
    for i in range(k):
        tmp_ar  = array[:,i*N_k:(i+1)*N_k]
        tmp_med = np.median(tmp_ar, axis=1)
        res_array.append(tmp_med)

    res_array   = np.array(res_array)
    return res_array.T

def remove_outlier(data, nsigma=4):
    med     = np.median(data[1])
    mad     = np.median(np.abs(data[1] - med))
    std     = 1.48*mad
    thres_p = med + std*nsigma
    thres_n = med - std*nsigma

    data_r  = copy.copy(data)
    data_r[1,((data[1]<thres_n)|(thres_p<data[1]))] = med
    return data

def remove_flare(data, nsigma=4):
    med     = np.median(data[1])
    mad     = np.median(np.abs(data[1] - med))
    std     = 1.48*mad
    thres_p = med + std*nsigma

    data_r  = copy.copy(data)
    data_r[1,(thres_p<data[1])] = med

    return data_r

def detrend_lc(data, npoint=10):
    data[1] = data[1] - np.median(data[1])
    d_tr    = exten3_lc(data)
    medd    = med_bin(d_tr, npoint)

    fn      = interpolate.interp1d(medd[0], medd[1], kind='cubic')
    trend   = fn(data[0])
    trend   = trend - np.median(trend)

    datad   = np.copy(data)
    datad[1]= data[1] - trend

    return datad

def median1d(arr, k):
    w = len(arr)
    idx = np.fromfunction(lambda i, j: i + j, (k, w), dtype="i8") - k // 2
    idx[idx < 0] = 0
    idx[idx > w - 1] = w - 1

    return np.median(arr[idx], axis=0)

def remove_lowest(data, num_of_d=2):
    data_r  = copy.copy(data)
    med     = np.median(data[1])
    for i in range(num_of_d):
        mind    = np.min(data_r[1])
        data_r[1,(data[1] == mind)]   = med

    return data_r

def bin_lc(data, num_of_d=100):

    i_sort  = np.argsort(data[0])
    data    = data[:,i_sort]

    num_of_bin = len(data[1])//num_of_d
    means   = []
    for i in range(num_of_bin):
        tmp_f   = np.mean(data[:,i*num_of_d:(i+1)*num_of_d], axis=1)
        means.append(tmp_f)
    means   = np.vstack(means)
    means   = means.T

    return means