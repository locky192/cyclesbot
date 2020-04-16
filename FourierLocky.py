import quandl
import concurrent.futures
import math
from numpy import arange, sin, pi
import numpy as np
import matplotlib.pyplot as plt

n_predict = 400
noisethrsh = 1/30
Ampthrsh = 10

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
Ffreq =    freq[np.logical_and(freq <= noisethrsh, Amp >= Ampthrsh)]
Fphase =     phase[np.logical_and(freq <= noisethrsh, Amp >= Ampthrsh)]
FAmp =       Amp[np.logical_and(freq <= noisethrsh, Amp >= Ampthrsh)]

freqmin = 1/800
freqmax = 1/200

PFfreq =    Ffreq[np.logical_and(Ffreq <= freqmax, Ffreq >= freqmin)]
phase =     Fphase[np.logical_and(Ffreq <= freqmax, Ffreq >= freqmin)]
Amp =       FAmp[np.logical_and(Ffreq <= freqmax, Ffreq >= freqmin)]

freqmin1 = 0
freqmax1 = 10000

PFfreq1 =    Ffreq[np.logical_and(Ffreq <= freqmax1, Ffreq >= freqmin1)]
phase1 =     Fphase[np.logical_and(Ffreq <= freqmax1, Ffreq >= freqmin1)]
Amp1 =       FAmp[np.logical_and(Ffreq <= freqmax1, Ffreq >= freqmin1)]

#recreate data using filterd waves
t = np.arange(0, n + n_predict)
signal = np.zeros(len(t))
for i in range(0, len(PFfreq)):
    signal += Amp[i] * np.cos(2 * np.pi * PFfreq[i] * t + phase[i])


#PFfreq = freq[freq >= noisethrsh and Amp >= Ampthrsh]
#Amp = Amp[freq >= noisethrsh and Amp >= Ampthrsh]

#filter low amplitude
fig, ax1 = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
ax1[0].set_yscale('log')
ax2 = ax1[0].twinx()
ax2.plot(np.arange(0, t.size), signal, 'r', label = 'extrapolation')
ax1[0].plot(np.arange(0, len(prices2)), prices2, 'b', label = 'x', linewidth = 3)
#plt.legend()
ax1[1].plot (1/PFfreq1, Amp1)
print(1/PFfreq[ Amp> 200])


#plt.figure()
#plt.plot(1/freq, np.angle(a) )
plt.show()
