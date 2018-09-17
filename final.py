import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import glob

import scipy.signal as sg

import pywt

path = "./fase_2/base_treinamento_II/"
path_test = "./fase_2/base_validacao_II/"

#path = "./fase_1/base_treinamento_I/"
#path_test = "./fase_1/base_validacao_I/"


def merge_intervals(intervals,thresh,mult_len,interval_thresh = 2000, interval_min = 500):
    #plt.vlines(intervals, 0, 2, color='green', linestyle='--',linewidth=3, alpha=0.5, label='Segment boundaries')
    result = []
    i = 1
    result.append(intervals[0])
    while(i<len(intervals)):
        #Verifica distancia entre as duas segmentacoes e se estao dentro dos 2 seg de cada letra
        calc = int((len(result)*mult_len))+1
        interval_distance = intervals[i-1][1] - intervals[i-1][0]
        if(interval_distance < interval_thresh and interval_distance > interval_min): # Itervalo tem que der uma distancia minima e um tam minimo
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


    y = lowPass(y)
    y = normalize(y)
    y = filterBySample(y, [1,200],cut_by_max=True)

    
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
    #plt.show()

    #plt.vlines(voiced_intervals, 0, 2, color='red', linestyle='--',linewidth=3, alpha=0.5, label='Segment boundaries')


    #merged = merge_intervals(voiced_intervals,4800,(len(y)-1)/4.0)
    merged = interval_time(len(y)-1,4)

    seg = [] 
    seg_filtered = []

    # splitting 
    for sg in merged:
        seg.append(y[sg[0]:sg[1]])

    #'''
    for sg in seg:
    
        sub_intervals = librosa.effects.split(sg, top_db=18, frame_length=1024, hop_length=100) #Tirando ruido remanescente
        
        #sub_intervals = librosa.effects.split(sg, top_db=20, frame_length=1024, hop_length=100) #USADO NA OUTRA ENTREGA
        sub_intervals = merge_intervals(sub_intervals,len(sg),1)
    
        interval = sub_intervals.pop()
        filtered = sg[interval[0]:interval[1]]
        seg_filtered.append(filtered)

        #seg_filtered.append(sg)

        #plt.figure(figsize=(12, 4))
        #plt.plot(sg)
        #plt.vlines(interval, 0, 0.1, color='red', linestyle='--',linewidth=3, alpha=0.5, label='Segment boundaries')
    
    #plt.show()
    #'''
    #seg_filtered = seg
    return seg_filtered, sr


def extract_features(segments,sr, n_mfcc = 20):
    features = []
    for segment in segments:
        #Calculandp mfcc
        mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc).T,axis=0)
        #tonnetz = librosa.feature.tonnetz(y=segment, sr=sr)

        #print(tonnetz)
        feat = mfcc

        #segment = np.array(segment)
        
        #Espelhando onda no eixo X
        #segment = abs(segment)

        #Contando quantidade de zeros
        #zeros = len(segment[segment == 0])
        #feat = np.append(feat,zeros)

        stz = stZCR(segment)
        ste = stEnergy(segment)
        stee = stEnergyEntropy(segment)
        feat = np.append(feat,stz)
        feat = np.append(feat,ste)
        feat = np.append(feat,stee)

        # Fourier
        segment = segment.T
        stft = np.abs(librosa.stft(segment))

        
        # chroma
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)

        # melspectrogram
        mel = np.mean(librosa.feature.melspectrogram(segment, sr=sr).T,axis=0)

        # spectral contrast
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)

        #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(segment), sr=sr).T,axis=0)

        feat = np.append(feat,chroma)
        feat = np.append(feat,mel)
        feat = np.append(feat,contrast)
        #feat = np.append(feat,tonnetz)
        

        #Calculando itens acima da media
        #mean = segment.mean()
        #over_mean = len(segment[segment >= mean])
        
        #feat = np.append(feat,over_mean)

        #Tamanho
        #total_len = len(segment[segment != 0])
        #feat = np.append(feat,total_len)

        features.append(feat)

    return features


def filterBySample(signal, sample, cut_by_max=False, percent_mean=0.7, max_noise = 0.15, default_noise=0.1):
    #print(len(signal))
    signal_sample = signal[sample[0]:sample[1]]
    signal_sample = signal_sample[signal_sample > 0]
    signal_sample = abs(signal_sample)
    noise_mean = signal_sample.mean()
    if(not(cut_by_max)):
        signal_sample = signal_sample[signal_sample > (noise_mean + noise_mean*percent_mean)]
        noise_mean = signal_sample.mean()
    else:
        noise_mean = signal_sample.max()

    #plt.figure(figsize=(12, 4))
    #plt.plot(signal)
    

    if (noise_mean > max_noise):
        noise_mean = default_noise
    
    #noise_mean = 1

    for x in np.nditer(signal, op_flags=['readwrite']) :
        if(x < 0):
            if(noise_mean < abs(x)):
                x += noise_mean
            else:
                x *= 0
            #print(x)
        else:
            if(x > noise_mean):
                x -= noise_mean
            else:
                x *= 0
    #plt.figure(figsize=(12, 4))
    #plt.plot(signal)
    

    return signal
    
def lowPass(signal):
    # First, design the Buterworth filter
    N  = 5    # 3, 4 Filter order
    Wn = 0.08 # 0.1 Cutoff frequency
    B, A = sg.butter(N, Wn, output='ba')
    smooth_data = sg.filtfilt(B,A, signal)
    #plt.plot(signal,'r-')
    #plt.plot(smooth_data,'b-')
    #plt.show()
    return smooth_data

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

# ===== Retirei de uma lib/paper : https://github.com/tyiannak/pyAudioAnalysis

eps = 0.00000001

def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return (np.float64(countZ) / np.float64(count-1.0))


def stEnergy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))


def stEnergyEntropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    Eol = np.sum(frame ** 2)    # total frame energy
    L = len(frame)
    sub_win_len = int(np.floor(L / n_short_blocks))
    if L != sub_win_len * n_short_blocks:
            frame = frame[0:sub_win_len * n_short_blocks]
    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (Eol + eps)

    # Compute entropy of the normalized sub-frame energies:
    Entropy = -np.sum(s * np.log2(s + eps))
    return Entropy


#main()