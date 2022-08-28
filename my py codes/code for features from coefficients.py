from scipy.signal import butter,filtfilt,freqz
import csv
import matplotlib.pyplot as plt
import numpy as np
import operator
import pywt
import xlsxwriter
import os
import numba as nb
from xlsxwriter import Workbook
from pyentrp import entropy as ent
import pyeeg
m=0









# Variables globales
nb_scales = 20
length_sample = 1000




## Coarse graining procedure
# tau : scale factor
# signal : original signal
# return the coarse_graining signal
def coarse_graining(tau, signal):
    # signal lenght
    N = len(signal)
    # Coarse_graining signal initialisation
    y = np.zeros(int(len(signal) / tau))
    for j in range(0, int(N / tau)):
        y[j] = sum(signal[i] / tau for i in range(int((j - 1) * tau), int(j * tau)))
    return y


## Multi-scale entropy
# m : length of the patterns that compared to each other
# r : tolerance
# signal : original signal
# return the Multi-scale entropy of the original signal (array of nbscales length)
def mse(signal,m,r, nbscales=None):
    # Output initialisation
    if nbscales == None:
        nbscales = int((len(signal) * nb_scales) / length_sample)
    y = np.zeros(nbscales + 1)
    y[0] = float('nan')
    for i in range(1, nbscales + 1):
        y[i] = pyeeg.samp_entropy(coarse_graining(i, signal), m, r)
    return y








@nb.jit(fastmath=True,error_model='numpy')
def sample_entropy(time_series, sample_length, tolerance=None):
    """Calculate and return Sample Entropy of the given time series.
    Distance between two vectors defined as Euclidean distance and can
    be changed in future releases
    Args:
        time_series: Vector or string of the sample data
        sample_length: Number of sequential points of the time series
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))
    Returns:
        Vector containing Sample Entropy (float)
    References:
        [1] http://en.wikipedia.org/wiki/mse
        [2] http://physionet.incor.usp.br/physiotools/sampen/
        [3] Madalena Costa, Ary Goldberger, CK Peng. Multiscale entropy analysis
            of biological signals
    """
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

    n = len(time_series)
    prev = np.zeros(n)
    curr = np.zeros(n)
    A = np.zeros((sample_length))  # number of matches for m = [1,...,template_length - 1]
    B = np.zeros((sample_length))  # number of matches for m = [1,...,template_length]

    for i in range(n - 1):
        nj = n - i - 1
        ts1 = time_series[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(time_series[j] - ts1) < tolerance:  # distance between two vectors
                curr[jj] = prev[jj] + 1
                temp_ts_length = min(sample_length, curr[jj])
                for m in range(int(temp_ts_length)):
                    A[m] += 1
                    if j < n - 1:
                        B[m] += 1
            else:
                curr[jj] = 0
        for j in range(nj):
            prev[j] = curr[j]

    N = n * (n - 1) // 2

    B2=np.empty(sample_length)
    B2[0]=N
    B2[1:]=B[:sample_length - 1]
    similarity_ratio = A / B2
    se = - np.log(similarity_ratio)
    return se





def butter_lowpass(cutoff, fs, order=6):     
    nyq = 0.5 * fs
    cut= cutoff/nyq
    b, a = butter(order,cut)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=6): #function to apply butterworth filter
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y



order=6     #butterworth filter order
fc = 60     #bandpass frequency in hz
fs = 512    #sample frequency in hz
wb1 = Workbook("mse from coeffs 1-750Ff.xlsx")

sheet1= wb1.add_worksheet('xmean')
sheet2= wb1.add_worksheet('ymean')
sheet3=wb1.add_worksheet('zmean')

path="D:\FN EEG Dataset\Data_F_Ind_1_750"
i=1
j=1


for filename in os.listdir(path):
    with open("D:\FN EEG Dataset\Data_F_Ind_1_750\{}".format(filename)) as f:                   #reading data from dataset using csv
         reader = csv.reader(f, delimiter=',')
         data = [[float(col1), float(col2)] for col1, col2 in reader]
    datx,daty=zip(*data)      #separating x and y from read data... datx and daty are tuples

    datax=list(datx)
    datay=list(daty)
    dataz=[]

    dataz=list(map(operator.sub,datax,datay))         #dataz=datax-datay
    x = butter_lowpass_filter(datax, fc, fs, order)   #applying butterworth filter to x
    y= butter_lowpass_filter(datay,fc,fs,order)       #to y
    z= butter_lowpass_filter(dataz,fc,fs,order)       #z=butterworth(x)-butterworth(y)


    coeffs = pywt.wavedec(x, 'db6', level=4)        #decomposition of 5 frequency bands...4th lvl and 6th order
    cA4,cD4,cD3,cD2,cD1=coeffs      #calculation of approximate and detailed coefficients for x values in dataset
    """
    cA4x=list(cA4)
    cD4x=list(cD4)
    cD3x=list(cD3)
    cD2x=list(cD2)
    cD1x=list(cD1)
    """
    shan=ent.shannon_entropy(np.round(cA4,2))
    std1=np.std(cA4)
    std2=np.std(cD4)
    std3=np.std(cD3)
    std4=np.std(cD2)
    std5=np.std(cD1)
    arr1=[mse(cA4,1,(0.2*std1)),mse(cD4,1,(0.2*std2)),mse(cD3,1,(0.2*std3)),mse(cD2,1,(0.2*std4)),mse(cD1,1,(0.2*std5))]
    
    
    coeffs = pywt.wavedec(y, 'db6', level=4)      
    cA4,cD4,cD3,cD2,cD1=coeffs
    """
    cA4y=list(cA4)
    cD4y=list(cD4)
    cD3y=list(cD3)
    cD2y=list(cD2)
    cD1y=list(cD1) 
    """
    shan=ent.shannon_entropy(np.round(cA4,2))
    std1=np.std(cA4)
    std2=np.std(cD4)
    std3=np.std(cD3)
    std4=np.std(cD2)
    std5=np.std(cD1)  
    arr2=[mse(cA4,1,(0.2*std1)),mse(cD4,1,(0.2*std2)),mse(cD3,1,(0.2*std3)),mse(cD2,1,(0.2*std4)),mse(cD1,1,(0.2*std5))]
    
    coeffs = pywt.wavedec(z, 'db6', level=4)    
    cA4,cD4,cD3,cD2,cD1=coeffs
    """
    cA4z=list(cA4)
    cD4z=list(cD4)
    cD3z=list(cD3)
    cD2z=list(cD2)
    cD1z=list(cD1)  
    """
    shan=ent.shannon_entropy(np.round(cA4,2))
    std1=np.std(cA4)
    std2=np.std(cD4)
    std3=np.std(cD3)
    std4=np.std(cD2)
    std5=np.std(cD1)
    arr3=[mse(cA4,1,(0.2*std1)),mse(cD4,1,(0.2*std2)),mse(cD3,1,(0.2*std3)),mse(cD2,1,(0.2*std4)),mse(cD1,1,(0.2*std5))]
    print(arr1[0])
    for z in range(len(arr1)):
        sheet1.write(i,j,arr1[z])
        sheet2.write(i,j,arr2[z])
        sheet3.write(i,j,arr3[z])
        j+=1       
    i+=1
    j=1

wb1.close()

