import quandl
import concurrent.futures
import math
from numpy import arange, sin, pi
import numpy as np
import matplotlib.pyplot as plt

n_predict = 400
noisethrsh = 1/5
Ampthrsh = 3

# Download latest BTC price data from quandl
quandl.ApiConfig.api_key = "dRsdc8njMS4QHeKqoJy-"
#print "Downloading BTC data..."
btcprice = quandl.get("BCHAIN/MKPRU", returns="numpy", collapse="daily")
#print "BTC data downloaded"

#split prices and dates into seperate lists
prices2 = []
dates = []
x = 0
for i in btcprice:
    if btcprice[x][1] != 0.0:
        dates.append(btcprice[x][0])
        prices2.append(btcprice[x][1])
    x = x + 1

n = np.size(prices2)

a = np.fft.fft(prices2)
phase = np.angle(a)
freq = np.fft.fftfreq(n)
#find amplitude of numpy's particular fft Algorithm
Amp = np.abs(a/n)

#filter 'noise' ie figh frequencies and low amplitude
PFfreq =    freq[np.logical_and(freq <= noisethrsh, Amp >= Ampthrsh)]
phase =     phase[np.logical_and(freq <= noisethrsh, Amp >= Ampthrsh)]
Amp =       Amp[np.logical_and(freq <= noisethrsh, Amp >= Ampthrsh)]


#recreate data using filterd waves
t = np.arange(0, n + n_predict)
signal = np.zeros(len(t))
for i in range(0, len(PFfreq)):
    signal += Amp[i] * np.cos(2 * np.pi * PFfreq[i] * t + phase[i])


#PFfreq = freq[freq >= noisethrsh and Amp >= Ampthrsh]
#Amp = Amp[freq >= noisethrsh and Amp >= Ampthrsh]

#filter low amplitude
plt.plot(np.arange(0, t.size), signal, 'r', label = 'extrapolation')
plt.plot(np.arange(0, len(prices2)), prices2, 'b', label = 'x', linewidth = 3)
plt.legend()
plt.show()
#plt.figure()
#plt.plot(1/freq, np.angle(a) )
plt.show()