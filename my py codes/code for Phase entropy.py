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
from scipy.signal import hilbert
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
   

    
    ren=[]
    i=0

    path="D:\project temp\csvs"
    for filename in os.listdir(path):
        print(filename)
        data=[]
        with open("D:\project temp\csvs\{}".format(filename)) as f:
            reader = csv.reader(f, delimiter=',') 
            data=list(reader)
        npdata=np.asarray(data)
        a=pyPTE.PTE(npdata)
        ren.append(a[1])
        i+=1
        print(i)
    Ren=np.transpose(ren)
    np.savetxt("result.csv", Ren[0], delimiter=",")
    np.savetxt("result2.csv",Ren[1],delimiter=",")
    np.savetxt("result3.csv",Ren[2],delimiter=",")

        
      
main()  



