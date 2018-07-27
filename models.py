import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

import random
from collections import Counter

import final as fi



# == Parametros
n_neighbors = 20
n_mfcc = 20 # 40
seed = 3

n_components = 150

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

mc_labels = ['a','b','c','d','h','m','n','x','6','7']

# == conjuntos 
# -Treino
X = []
y = []
# - Teste
Xt = []
yt = []

# Semente do random
random.seed(3)

print("\n#ETAPA 1 -  Lendo dataset e retirando features")
print("\nLendo dados de TREINO")
files = fi.getFiles(fi.path)
#files = files[:10]
random.shuffle(files)
i = 0
tam = len(files)
for file in files:
    
    #Debug
    if(i%10 == 0 ): print("Treino",i," de ",tam)
    
    segments,sr = fi.segment(file)
    features = fi.extract_features(segments,sr,n_mfcc=n_mfcc)
    X += features
    y += fi.getLabels(file)
    i+=1

X = np.array(X)#[8:])
y = np.array(y)#[8:])


print("\nLendo dados de TESTE")
files = fi.getFiles(fi.path_test)
random.shuffle(files)
#files = files[:10]
i = 0
tam = len(files)
for file in files:
    #Debug
    if(i%10 == 0 ): print("Teste",i," de ",tam)
    
    segments,sr = fi.segment(file)
    features = fi.extract_features(segments,sr,n_mfcc=n_mfcc)
    #Xt += features
    #yt += fi.getLabels(file)
    Xt.append(features)
    yt.append(fi.getLabels(file))
    i+=1


Xt = np.array(Xt)#[:8])
yt = np.array(yt)#[:8])

# Definindo modelos
print("\n#ETAPA 2 - Modelos")
print("\nInicalizando modelos")

models_name= [
    "RANDOM FOREST",
    "KNN - distance",
    "KNN - uniform",
    "Naive Bayes",
    "SVM",
    "Gaussian Process",
    "Neural Net"
]

models = [
    RandomForestClassifier(n_jobs=4, random_state=1),
    neighbors.KNeighborsClassifier(n_neighbors, weights='distance'),
    neighbors.KNeighborsClassifier(n_neighbors, weights='uniform'),
    GaussianNB(),
    GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid), #PCA(svd_solver='randomized',whiten=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    MLPClassifier(max_iter=500,hidden_layer_sizes=(32,16),activation='relu'),
    ]


# Selecionando modelos e setando pesos relativos
print("Selecionando modelos")

selected_models = [0,1,4,5]
relative_wheigths = [4,1,3,3]

aux1 = []
aux2 = []
for i in selected_models:
    aux1.append(models_name[i])
    aux2.append(models[i])

models_name = aux1
models = aux2

# Treinando os modelos

print("\n#ETAPA 3 - Treinamento")

i = 0
print("Treinando modelos")
for model in models:
    print("\nTreinando: "+models_name[i])
    model.fit(X,y)
    i+=1

# Predicao do teste

print("\n#ETAPA 4 - Teste/Validação")

j = 0
print("Predizendo - Final")
i = 0
captcha_solveds = 0
success = 0
statistics = [[],[]] # [pred, labels]

for ytn in yt:
    i = 0
    predicteds = [[],[],[],[]]
    result = []
 
     #debug
    if(j%10 == 0): print("Preditos:",j,"/",len(yt))
 
    for model in models:
        #print("\nPredizendo: "+models_name[i])
        p = model.predict(Xt[j])
        for k in range(4):
            #predicteds[k].append(p[k])
            #if(not(i == 0 and p[k] == 'b')):
            #    predicteds[k].append(p[k])
            for w in range(relative_wheigths[i]): # Peso relativo
                predicteds[k].append(p[k])
        i+=1
    #print(predicteds)
    #Escolhendo
    for k in range(4):
        if(predicteds[k][0] == 'b'): # Random forest tem mta colizao com b - se for desempata com outros modelos
            aux = predicteds[k][relative_wheigths[0]:]
            ct = Counter(aux)
            ct = ct.most_common()[0][0]
            result.append(ct)
        else: # se nao for 'b', random forest manda
         result.append(predicteds[k][0])   
        #ct = Counter(predicteds[k])
        #ct = ct.most_common()[0][0]
        #result.append(ct)
    
    result = np.array(result)
    c = ytn[ytn == result]
    success += len(c)
    if(len(c) == 4): captcha_solveds +=1
    #print(j)
    j+=1

# Printando resultados
print("\n#ETAPA 5 - RESULTADOS: ")
print("Caracteres corretos: ",success,"de",len(yt)*4)
print("Captchas corretos: ",captcha_solveds,"de",len(yt))



'''
# Código utilizado para selecao dos modelos
# ==== ANALISANDO MODELOS

j = 0
print("Predizendo")
for model in models:
    i = 0
    captcha_solveds = 0
    success = 0
    statistics = [[],[]] # [pred, labels]
    print("\nPredizendo: "+models_name[j])
    for ytn in yt:
        p = model.predict(Xt[i])
        c = ytn[ytn == p]
        success += len(c)
        if(len(c) == 4): captcha_solveds +=1
        statistics[0] = np.append(statistics[0], p)
        statistics[1] = np.append(statistics[1],ytn)
        
        i+=1
    #print(statistics)
    print("Caracteres corretos: ",success,"de",len(yt)*4)
    print("Captchas corretos: ",captcha_solveds,"de",len(yt))
    print("Matriz de confusao")
    print(mc_labels)
    print(confusion_matrix(statistics[1], statistics[0],labels = mc_labels))
    j+=1
'''




#print("\n\nTREINO ",y)
#print("\nTESTE: ",yt)

