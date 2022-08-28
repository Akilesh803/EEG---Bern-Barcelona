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
from scipy.stats import skew,kurtosis,iqr
m=0




def butter_lowpass(cutoff, fs, order=6):     
    nyq = 0.5 * fs
    cut= cutoff/nyq
    b, a = butter(order,cut)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=6): #function to apply butterworth filter
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

@jit(fastmath=True,error_model="numpy")
def feat(a4,d4,d3,d2,d1):
    
    arr=[skew(a4),skew(d4),skew(d3),skew(d2),skew(d1),kurtosis(a4),kurtosis(d4),kurtosis(d3),kurtosis(d2),kurtosis(d1),np.median(a4),np.median(d4),np.median(d3),np.median(d2),np.median(d1),np.sum(np.square(a4)),np.sum(np.square(d4)),np.sum(np.square(d3)),np.sum(np.square(d2)),np.sum(np.square(d1))]
    return arr


def main():
    order=6     #butterworth filter order
    fc = 60     #bandpass frequency in hz
    fs = 512    #sample frequency in hz
    wb1 = Workbook("additional features from coeffs 3000-3750nf.xlsx")

    sheet1= wb1.add_worksheet('xmean')
    sheet2= wb1.add_worksheet('ymean')
    sheet3=wb1.add_worksheet('zmean')

    path="D:\FN EEG Dataset\Data_N_Ind_3001_3750"
    i=1
    j=1
    for filename in os.listdir(path):
        with open("D:\FN EEG Dataset\Data_N_Ind_3001_3750\{}".format(filename)) as f:
            reader = csv.reader(f, delimiter=',') 
            data = [[float(col1), float(col2)] for col1, col2 in reader]
        datax,datay=zip(*data)      #separating x and y from read data... datx and daty are tuples

        

        dataz=list(map(operator.sub,datax,datay))         #dataz=datax-datay
        x = butter_lowpass_filter(datax, fc, fs, order)   #applying butterworth filter to x
        y= butter_lowpass_filter(datay,fc,fs,order)       #to y
        z= butter_lowpass_filter(dataz,fc,fs,order)       #z=butterworth(x)-butterworth(y)

    
        coeffs = pywt.wavedec(x, 'db6', level=4)        #decomposition of 5 frequency bands...4th lvl and 6th order
        cA4,cD4,cD3,cD2,cD1=coeffs      #calculation of approximate and detailed coefficients for x values in dataset
    
        cA4x=list(cA4)
        cD4x=list(cD4)
        cD3x=list(cD3)
        cD2x=list(cD2)
        cD1x=list(cD1)
        
        arr1=feat(cA4x,cD4x,cD3x,cD2x,cD1x)
        
        coeffs = pywt.wavedec(y, 'db6', level=4)      
        cA4,cD4,cD3,cD2,cD1=coeffs
    
        cA4y=list(cA4)
        cD4y=list(cD4)
        cD3y=list(cD3)
        cD2y=list(cD2)
        cD1y=list(cD1) 
    
        
        arr2=feat(cA4y,cD4y,cD3y,cD2y,cD1y)
        
        coeffs = pywt.wavedec(z, 'db6', level=4)    
        cA4,cD4,cD3,cD2,cD1=coeffs
    
        cA4z=list(cA4)
        cD4z=list(cD4)
        cD3z=list(cD3)
        cD2z=list(cD2)
        cD1z=list(cD1)  
    
        
        arr3=feat(cA4z,cD4z,cD3z,cD2z,cD1z)
        print(i)
        for _ in range(len(arr1)):
            sheet1.write(i,j,arr1[_])
            sheet2.write(i,j,arr2[_])
            sheet3.write(i,j,arr3[_])
            j+=1       
        i+=1
        j=1
    wb1.close()

main()  



