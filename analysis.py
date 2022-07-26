#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

import sys

import getlc
import fft


epicid  = "210721261"

lck2    = getlc.k2lc_byepic(epicid)
lctess  = getlc.tesslc_byepic(epicid)


test_lc     = fft.lowpass_filter(lck2,10)

plt.plot(test_lc[0]+lck2[0,0], test_lc[1])
plt.plot(lck2[0], lck2[1])
plt.show()
