import quandl
import concurrent.futures
import math
from numpy import arange, sin, pi
import numpy as np
import matplotlib.pyplot as plt

n_predict = 10 #amount of days to predict in the future
noisethrsh = 0
Ampthrsh = 0
LookBackPeriod = 100 # number of days the data is based on 


# Download latest BTC price data from quandl
quandl.ApiConfig.api_key = "dRsdc8njMS4QHeKqoJy-"
#print "Downloading BTC data..."
btcprice = quandl.get("BCHAIN/MKPRU", returns="numpy", collapse="daily")
#print "BTC data downloaded"

#split prices and dates into seperate lists
prices1 = []
dates = []
x = 0
for i in btcprice:
    if btcprice[x][1] != 0.0:
        dates.append(btcprice[x][0])
        prices1.append(btcprice[x][1])
    x = x + 1
prices1 = prices1 #[0:1000] #make smaller so faster to run 
priceslog  = np.log(np.array(prices1)*100) #turns the data to log form, x100 so not decimials therefor no -ve numbers 

FowardSignal = []
i=0
while  i + LookBackPeriod < (len(prices1)):
    prices2 = priceslog[i : i+LookBackPeriod] 
    n = np.size(prices2)
    a = np.fft.fft(prices2)
    phase = np.angle(a)
    freq = np.fft.fftfreq(n)
    #find amplitude of numpy's particular fft Algorithm
    Amp = np.abs(a/n)

    #filter 'noise' i.e figh frequencies and low amplitude
    PFfreq =    freq[np.logical_and(freq <= noisethrsh, Amp >= Ampthrsh)] #get rid of high freq low amp and correspond phase +amp
    phase =     phase[np.logical_and(freq <= noisethrsh, Amp >= Ampthrsh)]
    Amp =       Amp[np.logical_and(freq <= noisethrsh, Amp >= Ampthrsh)]


    #recreate data using filterd waves
    t = n + n_predict #predicts n + n_perdict value in the futuyre 
    signal = np.zeros(1)
    for x in range(0, len(PFfreq)):
        signal += Amp[x] * np.cos(2 * np.pi * PFfreq[x] * t + phase[x])
    FowardSignal = np.append(FowardSignal, signal)
    i+=1 
#
#




#filter low amplitude
plt.plot(np.arange(n_predict, n_predict  + FowardSignal.size), FowardSignal, 'r', label = 'extrapolation')
plt.plot(np.arange(0, len(priceslog)), priceslog, 'b', label = 'x', linewidth = 3)
plt.legend()
plt.show()

plt.show()

#linear detrend 

