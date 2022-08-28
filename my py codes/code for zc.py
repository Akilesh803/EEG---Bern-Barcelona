from scipy.signal import butter,filtfilt,freqz
import csv
import matplotlib.pyplot as plt
import numpy as np
import operator
import pywt
import xlsxwriter
import os
import numba as nb
from numba import jit
from xlsxwriter import Workbook
from pyentrp import entropy as ent
import pyeeg as pg
import math
from scipy.stats import skew,kurtosis,iqr
from matplotlib.mlab import find
from pyPTE import pyPTE
m=0

def mode(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m]






def butter_lowpass(cutoff, fs, order=6):     
    nyq = 0.5 * fs
    cut= cutoff/nyq
    b, a = butter(order,cut)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=6): #function to apply butterworth filter
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def freq_from_crossings(sig, fs):
    """
    Estimate frequency by counting zero crossings
    """
    # Find all indices right before a rising-edge zero crossing
    indices = find((sig[1:] >= 0) & (sig[:-1] < 0))

    # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
    # crossings = indices

    # More accurate, using linear interpolation to find intersample
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]

    # Some other interpolation based on neighboring points might be better.
    # Spline, cubic, whatever

    return fs / np.mean(np.diff(crossings))




def zero_crossing(data):
        return (((data[:-1] * data[1:]) < 0).sum())

def renyi_entropy(data,alpha):
    Ren=[]
    #make sure to round the signal points to integer
    
    #data=np.around(data)
    #iterate each signal
    for i in range(3750):
        X=data[i]
        data_set = list(set(X))
        freq_list = []
        for entry in data_set:
            counter = 0.
            for i in X:
                if i == entry:
                    counter += 1
            freq_list.append(float(counter)/len(X))
        summation=0
        for freq in freq_list:
            summation+=math.pow(freq,alpha)
        Ren.append((1/float(1-alpha))*(math.log(summation,2)))

    return Ren



def tsallis_entropy(data,alpha):
    
    #make sure to round the signal points to integer
    Tsa=[]
    
    #iterate each signal
    for i in range(3750):
        X=data[i]
        data_set = list(set(X))
        freq_list = []
        for entry in data_set:
            counter = 0.
            for i in X:
                if i == entry:
                    counter += 1
            freq_list.append(float(counter)/len(X))
        summation=0
        for freq in freq_list:
            summation+=math.pow(freq,alpha)
        Tsa.append((1/float(alpha-1))*(1-summation))
    return Tsa



def main():
   

    order=6     #butterworth filter order
    fc = 60     #bandpass frequency in hz
    fs = 512    #sample frequency in hz
    
    ren=[]
    i=0
    res1=[]
    res2=[]
    res3=[]
    path="D:\project\FN EEG Dataset\Data_F_Ind_1_750"
    for filename in os.listdir(path):
        print(filename)
        data=[]
        with open("D:\project\FN EEG Dataset\Data_F_Ind_1_750\{}".format(filename)) as f:
            reader = csv.reader(f, delimiter=',') 
            data = [[float(col1), float(col2)] for col1, col2 in reader]
        datax,datay=zip(*data) 
        dataz=list(map(operator.sub,datax,datay))         #dataz=datax-datay
        x = butter_lowpass_filter(datax, fc, fs, order)   #applying butterworth filter to x
        y= butter_lowpass_filter(datay,fc,fs,order)       #to y
        z= butter_lowpass_filter(dataz,fc,fs,order)
        i+=1
        print(freq_from_crossings(x,512))
        #res1.append(zero_crossing(x))
        #res2.append(zero_crossing(y))
        #res3.append(zero_crossing(z))
    
    #np.savetxt("zc 3000-3750nf x.csv", res1, delimiter=",")
    #np.savetxt("zc 3000-3750nf y.csv",res2,delimiter=",")
    #np.savetxt("zc 3000-3750nf z.csv",res3,delimiter=",")

        
      
main()  



