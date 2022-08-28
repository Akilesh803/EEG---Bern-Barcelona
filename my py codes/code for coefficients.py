from scipy.signal import butter,filtfilt,freqz
import csv
import matplotlib.pyplot as plt
import numpy as np
import operator
import pywt
import xlsxwriter
import os
from xlsxwriter import Workbook
from pyentrp import entropy as ent

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



order=6     #butterworth filter order
fc = 60     #bandpass frequency in hz
fs = 512    #sample frequency in hz
wb1 = Workbook("cA4 1-750f.xlsx")
wb2= Workbook("cD4 1-750f.xlsx")
wb3=Workbook("cD3 1-750f.xlsx")
wb4=Workbook("cD2 1-750f.xlsx")
wb5=Workbook("cD1 1-750f.xlsx")
sheet11= wb1.add_worksheet('xmean')
sheet12= wb1.add_worksheet('ymean')
sheet13=wb1.add_worksheet('zmean')
sheet21= wb2.add_worksheet('xmean')
sheet22= wb2.add_worksheet('ymean')
sheet23=wb2.add_worksheet('zmean')
sheet31= wb3.add_worksheet('xmean')
sheet32= wb3.add_worksheet('ymean')
sheet33=wb3.add_worksheet('zmean')
sheet41= wb4.add_worksheet('xmean')
sheet42= wb4.add_worksheet('ymean')
sheet43=wb4.add_worksheet('zmean')
sheet51= wb5.add_worksheet('xmean')
sheet52= wb5.add_worksheet('ymean')
sheet53=wb5.add_worksheet('zmean')
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
    
    std1=np.std(cA4)
    std2=np.std(cD4)
    std3=np.std(cD3)
    std4=np.std(cD2)
    std5=np.std(cD1)
    arr1=[np.mean(cA4),np.mean(cD4),np.mean(cD3),np.mean(cD2),np.mean(cD1),np.min(cA4),np.min(cD4),np.min(cD3),np.min(cD2),np.min(cD1),np.max(cA4),np.max(cD4),np.max(cD3),np.max(cD2),np.max(cD1),std1,std2,std3,std4,std5,shan,ent.shannon_entropy(np.round(cD4,2)),ent.shannon_entropy(np.round(cD3,2)),ent.shannon_entropy(np.round(cD2,2)),ent.shannon_entropy(np.round(cD1,2)),ent.permutation_entropy(cA4,3,1),ent.permutation_entropy(cD4,3,1),ent.permutation_entropy(cD3,3,1),ent.permutation_entropy(cD2,3,1),ent.permutation_entropy(cD1,3,1),ent.sample_entropy(cA4,1,0.2*std1),ent.sample_entropy(cD4,1,0.2*std2),ent.sample_entropy(cD3,1,0.2*std3),ent.sample_entropy(cD2,1,0.2*std4),ent.sample_entropy(cD1,1,0.2*std5)]
    
    
    coeffs = pywt.wavedec(y, 'db6', level=4)      
    cA4,cD4,cD3,cD2,cD1=coeffs
    """
    cA4y=list(cA4)
    cD4y=list(cD4)
    cD3y=list(cD3)
    cD2y=list(cD2)
    cD1y=list(cD1) 
    """
    shan=ent.shannon_entropy(cA4)
    std1=np.std(cA4)
    std2=np.std(cD4)
    std3=np.std(cD3)
    std4=np.std(cD2)
    std5=np.std(cD1)  
    arr2=[np.mean(cA4),np.mean(cD4),np.mean(cD3),np.mean(cD2),np.mean(cD1),std1,std2,std3,std4,std5,shan,ent.shannon_entropy(cD4),ent.shannon_entropy(cD3),ent.shannon_entropy(cD2),ent.shannon_entropy(cD1),ent.permutation_entropy(cA4,3,1),ent.permutation_entropy(cD4,3,1),ent.permutation_entropy(cD3,3,1),ent.permutation_entropy(cD2,3,1),ent.permutation_entropy(cD1,3,1),ent.sample_entropy(cA4,1,0.2*std1),ent.sample_entropy(cD4,1,0.2*std2),ent.sample_entropy(cD3,1,0.2*std3),ent.sample_entropy(cD2,1,0.2*std4),ent.sample_entropy(cD1,1,0.2*std5)]   
    
    coeffs = pywt.wavedec(z, 'db6', level=4)    
    cA4,cD4,cD3,cD2,cD1=coeffs
    """
    cA4z=list(cA4)
    cD4z=list(cD4)
    cD3z=list(cD3)
    cD2z=list(cD2)
    cD1z=list(cD1)  
    """
    shan=ent.shannon_entropy(cA4)
    std1=np.std(cA4)
    std2=np.std(cD4)
    std3=np.std(cD3)
    std4=np.std(cD2)
    std5=np.std(cD1)
    arr3=[np.mean(cA4),np.mean(cD4),np.mean(cD3),np.mean(cD2),np.mean(cD1),std1,std2,std3,std4,std5,shan,ent.shannon_entropy(cD4),ent.shannon_entropy(cD3),ent.shannon_entropy(cD2),ent.shannon_entropy(cD1),ent.permutation_entropy(cA4,3,1),ent.permutation_entropy(cD4,3,1),ent.permutation_entropy(cD3,3,1),ent.permutation_entropy(cD2,3,1),ent.permutation_entropy(cD1,3,1),ent.sample_entropy(cA4,1,0.2*std1),ent.sample_entropy(cD4,1,0.2*std2),ent.sample_entropy(cD3,1,0.2*std3),ent.sample_entropy(cD2,1,0.2*std4),ent.sample_entropy(cD1,1,0.2*std5)]
    
  """  
    for z in range(len(cA4x)):
        sheet11.write(i,j,cA4x[z])
        sheet12.write(i,j,cA4y[z])
        sheet13.write(i,j,cA4z[z])
        j+=1       
    
    j=1
    for z in range(len(cD4x)):
        sheet21.write(i,j,cD4x[z])
        sheet22.write(i,j,cD4y[z])
        sheet23.write(i,j,cD4z[z])
        j+=1       
    
    j=1
    for z in range(len(cD3x)):
        sheet31.write(i,j,cD3x[z])
        sheet32.write(i,j,cD3y[z])
        sheet33.write(i,j,cD3z[z])
        j+=1       
    
    j=1
    for z in range(len(cD2x)):
        sheet41.write(i,j,cD2x[z])
        sheet42.write(i,j,cD2y[z])
        sheet43.write(i,j,cD2z[z])
        j+=1       
   
    j=1
    for z in range(len(cD1x)):
        sheet51.write(i,j,cD1x[z])
        sheet52.write(i,j,cD1y[z])
        sheet53.write(i,j,cD1z[z])
        j+=1       
    
    j=1

    i+=1
"""
for z in range(len(arr1)):
        sheet1.write(i,j,arr1[z])
        sheet2.write(i,j,arr2[z])
        sheet3.write(i,j,arr3[z])
        j+=1       
    i+=1
    j=1
wb1.close()
wb2.close()
wb3.close()
wb4.close()
wb5.close()
