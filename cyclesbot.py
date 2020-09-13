import quandl
import concurrent.futures
import math
from numpy import arange, sin, pi
import numpy as np
import matplotlib.pyplot as plt
import csv
import os.path
from os import path
from Tkinter import *
import ttk
from scipy.optimize import curve_fit
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 

## SET MAX WIDTH HERE
maxwidth = 100

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

prices = prices2#[0:2050]

def createsin(width, lenth):
    x = 0
    wave = []
    for i in range(lenth): # maximum number of data points
        x = x + math.pi/width
        wave.append(sin(x))
    return wave

def shiftleft(wavelist,num):
    for i in range(num):
        wavelist.pop(0)

#scan to find cycles with high correlation
def scan(width):
    x = 0
    amplitude = []

    while x < 2*width:
        sine = createsin(width, len(prices)+(2*width))
        for i in range(x):
            sine.pop()
        shiftleft(sine,2*width-x)
        amplitude.append(np.corrcoef(sine,prices)[0][1])
        x = x +1

    return [max(amplitude), amplitude.index(max(amplitude)), width]

def EMA(bandlength, close_list_name):
    #close_list_name = map(float, close_list_name)
    emalist = []
    emalist.append(close_list_name[-1])
    x = (len(close_list_name) - 1)
    k = (2.0 / (float(bandlength) + 1.0))

    for i in range(len(close_list_name)):
        emalist.append((close_list_name[x] * k) + (emalist[-1] * (1 - k)))
        x = x - 1
    emalist.reverse()

    del emalist[-1]
    del emalist[-1]

    return emalist

def average(waveslist):
    listlength = []
    for i in waveslist:
        listlength.append(len(i))

    minlength = min(listlength)

    x = 0
    i = 0
    n = 0
    q = 0
    ave = []
    while i < minlength:
        while x < len(waveslist):
            n = n + (waveslist[x][i])
            x = x + 1
        n = n/len(waveslist)
        ave.append(n)
        i = i + 1
        x = 0
        n = 0
    return ave

def bigScan():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        widthlist = range(2,maxwidth)
        x = 2
        tamp = []
        tindexs = []
        twidths = []
        results = [executor.submit(scan, width) for width in widthlist]

        for f in concurrent.futures.as_completed(results):
            print f.result()[2], "/", maxwidth
            tamp.append(f.result()[0])
            tindexs.append(f.result()[1])
            twidths.append(f.result()[2])
            x = x + 1
            
    amp = []
    indexs = []
    widths = []

    x = 2
    while x < len(tamp):
        i = twidths.index(x)
        amp.append(tamp[i])
        indexs.append(tindexs[i])
        widths.append(twidths[i])
        x = x + 1

    x = 0
    with open('output.csv','wb') as result_file:
        wr=csv.writer(result_file,dialect='excel')
        for i in amp:
            wr.writerow([amp[x],indexs[x],widths[x]])
            x = x + 1
            
def useData():
    val = raw_input("Existing Cycle Data Detected. Do you want to use this data? (Y/N)")
    if val == "N":
        bigScan()
    if val != "N" and val != "Y":
        print "Invalid Input"
        useData()

def func(x,a,b):
    return a*np.log(x)+ b

def Readstatus(key, count, key2):
    for i in root.winfo_children():
        if i.winfo_class() == 'Canvas' or i.winfo_class() == 'Frame':
            i.destroy() 
    
    yaCycles = []
    
    for i in range(len(tp)):    
        var_obj = var.get(i)  
        #print i, var_obj.get(), tp[i][0]
        if var_obj.get() == 1:
            specificCycle = createsin(tp[i][0],len(prices2*2))
            shiftleft(specificCycle,tp[i][1])
            yaCycles.append(specificCycle)

    eCycles = []

    for i in range(len(tp)):    
        var_obj = var2.get(i)  
        if var_obj.get() == 1:
            specificCycle = createsin(tp[i][0],len(prices2*2))
            shiftleft(specificCycle,tp[i][1])
            eCycles.append(specificCycle)

    var_obj = var.get('UD')
    if len(yaCycles) > 0:
        if var_obj.get() == 0:
            Flist = average(yaCycles)
        else:
            specificCycle = createsin(20000,len(prices2*2))
            shiftleft(specificCycle,1000)
            Flist = average([average(yaCycles),specificCycle])
        if len(eCycles) > 0:
            Flist = average([Flist,average(eCycles)])
    else:
        if var_obj.get() == 1:
            specificCycle = createsin(20000,len(prices2*2))
            shiftleft(specificCycle,1000)
            Flist = specificCycle
        else:
            Flist = []
##    xdata=range(1,len(prices2[500:-1])+1)
##    #print xdata
##    ydata=prices2[500:-1]
##    #print ydata
##
##    x = np.array(xdata, dtype=float) #transform your data in a numpy array of floats 
##    y = np.array(ydata, dtype=float) #so the curve_fit can work
##
##    popt, pcov = curve_fit(func, x, y)
##    x_sorted = np.sort(x)
##    #Flist = func(x, *popt)
    
    fig, ax1 = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    ax1[0].set_yscale('log')
    ax1[0].set_ylim([1,5000000])
    #ax1[1].set_ylim([-0.3,0.6])
    ax1[1].plot(amp, color="blue")
    ax1[1].scatter(xlist,ylist, color='r')
    ax1[1].scatter(xsmall,ysmall, color='green')
    ax1[0].plot(prices2)
    ax2 = ax1[0].twinx()
    #ax1[1].set_yscale('log')
    #ax2.set_ylim([0.01,1])
    ax2.plot(Flist, color="red") 
  
    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, master = root)    
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().grid(row = 2, rowspan=count, column = 0, pady = 2) 
  
    # creating the Matplotlib toolbar 
    #toolbar = NavigationToolbar2Tk(canvas, root) 
    #toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    #canvas.get_tk_widget().grid(row = 3, column = 0, pady = 2)

if __name__ == '__main__':
    
    savedData = path.exists("output.csv")

    if savedData == False:
        print "No Existing Data Detected. Starting cycles scan."
        bigScan()
    else:
        useData()
        
    with open('output.csv', 'rb') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    amp = []
    indexs = []
    widths = []
    
    for i in your_list:
        amp.append(float(i[0]))
        indexs.append(int(i[1]))
        widths.append(int(i[2]))
        
    #find key long term cycles

    tpamp = []
    tp = []
    y = len(amp) - 5
    while y > 5:
        if amp[y] > amp[y-1] and amp[y] > amp[y-2] and amp[y] > amp[y+1] and amp[y+2] and amp[y] > amp[y-3] and amp[y] > amp[y+3] and amp[y] > amp[y-4] and amp[y] > amp[y+4]:
            print y, amp[y], indexs[y]
            tpamp.append(amp[y])
            tp.append([y,indexs[y]])
        y = y - 1

    keylist = []
    ulist = []
    x = 1
    llist = []
    while x < len(tpamp):
        if tpamp[x-1] < tpamp[x] and tpamp[x] > tpamp[x+1]:
            keylist.append([tpamp[x], tp[x]])
        else:
            if len(keylist) < 2:
                ulist.append([tpamp[x],tp[x]])
            else:
                llist.append([tpamp[x],tp[x]])
                        
        x = x + 1

    index = tpamp.index(max(tpamp))
    
    keylist.append([tpamp[index], tp[index]])

    famp = []
    
    for p in ulist:
        keylist.append(p)
        xlist = []
        ylist = []
        j = 0
        ilist = []
        customcycles = []
        for i in keylist:
            if j < 3:
                xlist.append(i[1][0])
                ylist.append(i[0])
                ilist.append(i[1][1])
            j = j + 1

        customcycles.append([[[20000],
                          [1000]],
                         [xlist,
                          ilist]])

        results = []

        for i in customcycles:
            for j in i:
                tresult = []
                x=0
                while x < len(j[0]):
                    sine = createsin(j[0][x], len(prices)+50000)#(2*widths[x])-50)
                    shiftleft(sine,2*j[0][x]-j[1][x])
                    tresult.append(sine)
                    x = x+1
                results.append(tresult)

        ilist = average([average(results[0]),average(results[1])])

        olist = EMA(1000,ilist)

        x = 0
        olist.reverse()
        while x < 78:
            olist.append(0)
            x = x + 1
        olist.reverse()

        while len(olist) > len(prices2):
            olist.pop()

        famp.append(np.corrcoef(olist,prices2)[0][1])
        keylist.pop()

    keylist.append(ulist[famp.index(max(famp))])
    xlist = []
    ylist = []
    xsmall = []
    ismall = []
    ysmall = []
    j = 0
    ilist = []
    customcycles = []
    print keylist
    for i in keylist:
        print i
        if j < 3:
            xlist.append(i[1][0])
            ylist.append(i[0])
            ilist.append(i[1][1])
        j = j + 1

    for i in llist:
        xsmall.append(i[1][0])
        ysmall.append(i[0])
        ismall.append(i[1][1])

    customcycles.append([[[20000],
                      [1000]],
                     [xlist,
                      ilist],
                         [xsmall,
                          ismall]])

    print customcycles

    results = []

    for i in customcycles:
        for j in i:
            tresult = []
            x=0
            while x < len(j[0]):
                sine = createsin(j[0][x], len(prices)+50000)#(2*widths[x])-50)
                shiftleft(sine,2*j[0][x]-j[1][x])
                tresult.append(sine)
                x = x+1
            results.append(tresult)

    ilist = average([average(results[0]),average(results[1])])

    olist = EMA(1000,ilist)

    x = 0
    olist.reverse()
    while x < 78:
        olist.append(0)
        x = x + 1
    olist.reverse()

    if len(results[2]) > 0:     
        plist = average([olist,average(results[2])])
    
        zlist = EMA(500, plist)

        x = 0
        zlist.reverse()
        while x < 90:
            zlist.append(0)
            x = x + 1
        zlist.reverse()
    else:
        zlist = olist

    while len(zlist) > len(prices2):
        zlist.pop()

    print
    print
    specificCycle = createsin(tp[0][0],10000)
    shiftleft(specificCycle,tp[0][1])


    #print tpamp

##    fig, ax1 = plt.subplots(2)
##    fig.suptitle('Vertically stacked subplots')
##    ax1[0].set_yscale('log')
##    ax1[0].set_ylim([1,5000000])
##    #ax1[1].set_ylim([-0.3,0.6])
##    ax1[1].plot(amp, color="blue")
##    ax1[1].scatter(xlist,ylist, color='r')
##    ax1[1].scatter(xsmall,ysmall, color='green')
##    ax1[0].plot(prices2)
##    ax2 = ax1[0].twinx()
##    #ax1[1].set_yscale('log')
##    #ax2.set_ylim([0.01,1])
##    ax2.plot(specificCycle, color="red")

    root = Tk()

    var = dict()

    var['UD'] = IntVar()
    upDrift = Checkbutton(root, text='upDrift', variable=var['UD'], 
                          command=lambda key='UD': Readstatus(key, count - 1, 0))
    upDrift.grid(row = 0, column = 1, pady = 2)

    premadeList = range(len(tp))
    
    count=1

    for i in premadeList:
        var[i]=IntVar()
        chk = Checkbutton(root, text=tp[i][0], variable=var[i], 
                          command=lambda key=i: Readstatus(key, count - 1, 0))
        count += 1
        chk.grid(row = count - 1, column = 1, pady = 2)

    var2 = dict()
    count=1
    for i in premadeList:
        var2[i]=IntVar()
        chk = Checkbutton(root, text=tp[i][0], variable=var2[i], 
                          command=lambda key=i: Readstatus(0, count -1, key))
        count += 1
        chk.grid(row = count - 1, column = 2, pady = 2)

    Readstatus(0, count - 1, 0)

    root.mainloop()
