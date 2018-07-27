import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import glob

import pywt

#file_name = './fase_1/base_treinamento_I/bhhn.wav'
#file_name = './fase_1/base_treinamento_I/xnmd.wav'
#file_name = './fase_1/base_treinamento_I/7bxa.wav'
#file_name = './fase_1/base_treinamento_I/6nxx.wav'
#file_name = './fase_1/base_treinamento_I/6n77.wav'

path = "./fase_1/base_treinamento_I/"
path_test = "./fase_1/base_validacao_I/"

def merge_intervals(intervals,thresh,mult_len,interval_thresh = 2000):
    result = []
    i = 1
    result.append(intervals[0])
    while(i<len(intervals)):
        #Verifica distancia entre as duas segmentacoes e se estao dentro dos 2 seg de cada letra
        calc = int((len(result)*mult_len))+1
        if(intervals[i-1][1] - intervals[i-1][0] < interval_thresh):
            result.pop()
            result.append(intervals[i])
        elif(intervals[i][0] - intervals[i-1][1] < thresh and (int(calc/intervals[i-1][1])) == (int(calc/intervals[i][1]))):
            aux = result.pop()
            result.append([aux[0],intervals[i][1]]) # junta os dois bounds
            #result.remove(intervals[i-1])
        else:
            #recalcula o corte virtual
            calc = int(((len(result)+1)*mult_len)) +1
            #Verifica se juntou dois audios na mesma seg
            if((int(calc/intervals[i][0])) != (int(calc/intervals[i][1]))):
                result.append([intervals[i][0],calc])
                result.append([calc,intervals[i][1]])
            else:
                if(intervals[i][1]-intervals[i][0] > 10):
                    result.append(intervals[i])
        i+=1
    return result


def interval_time(size,slices):
    result = []
    mult = int(size/slices)
    i = 1
    while(i <= slices):
        result.append([(mult*(i-1))+1, (mult*(i))-1])
        i+=1
    return result


def segment(file_name):
    #print("Segmentando: ",file_name)
    y, sr = librosa.load(file_name, mono =True)#, duration=8

    #https://librosa.github.io/librosa/generated/librosa.effects.split.html

    y = normalize(y)
    #y = wavelet(y)
    #y = waveletn(y,'coif1')# haar
    #y = librosa.util.normalize(y)

    #voiced_intervals = librosa.effects.split(y, top_db=30, frame_length=1024, hop_length=80)
    #voiced_intervals = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=100)
    #voiced_intervals = librosa.effects.split(y, top_db=20, frame_length=1024, hop_length=100)
    
    #voiced_intervals = librosa.effects.split(y, top_db=36, frame_length=2048, hop_length=100)
    
    #voiced_intervals = euristcSegment(y,4,2000,f_tollerance=0.8)
    
    #plt.figure(figsize=(12, 4))
    #plt.plot(y)
    #plt.vlines(voiced_intervals, 0, 2, color='red', linestyle='--',linewidth=3, alpha=0.5, label='Segment boundaries')


    #merged = merge_intervals(voiced_intervals,4800,(len(y)-1)/4.0)
    merged = interval_time(len(y)-1,4)

    seg = [] 
    seg_filtered = []

    # splitting 
    for sg in merged:
        seg.append(y[sg[0]:sg[1]])

    for sg in seg:
    
        #sub_intervals = librosa.effects.split(sg, top_db=25, frame_length=1024, hop_length=100) #Tirando ruído remanescente
        sub_intervals = librosa.effects.split(sg, top_db=20, frame_length=1024, hop_length=100) #Tirando ruído remanescente
        sub_intervals = merge_intervals(sub_intervals,len(sg),1)
    
        interval = sub_intervals.pop()
        filtered = sg[interval[0]:interval[1]]
        seg_filtered.append(filtered)

        #plt.figure(figsize=(12, 4))
        #plt.plot(sg)
        #plt.vlines(interval, 0, 2, color='red', linestyle='--',linewidth=3, alpha=0.5, label='Segment boundaries')
    
    #plt.show()

    return seg_filtered, sr


def extract_features(segments,sr, n_mfcc = 20):
    features = []
    for segment in segments:
        #Calculandp mfcc
        mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc).T,axis=0)

        feat = mfcc

        segment = np.array(segment)
        
        #Espelhando onda no eixo X
        segment = abs(segment)

        #Contando quantidade de zeros
        zeros = len(segment[segment == 0])
        #features.append(zeros)
        feat = np.append(feat,zeros)

        #Calculando itens acima da media
        mean = segment.mean()
        over_mean = len(segment[segment >= mean])
        #features.append(over_mean)
        feat = np.append(feat,over_mean)

        #Tamanho
        #total_len = len(segment[segment != 0])
        #feat = np.append(feat,total_len)

        features.append(feat)

    return features


def normalize(signal,pre_emphasis= 0.97):
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal

def wavelet(signal):
    return pywt.dwt(signal, 'db1')[1]

def waveletn(signal,wav):
    return pywt.dwt(signal, wav)[1]

def getFiles(path):
    files = glob.glob(path+"*.wav")
    return files

def getLabels(path_file):
    return list(path_file[-8:-4])

def main():
    getFiles(path)


#main()