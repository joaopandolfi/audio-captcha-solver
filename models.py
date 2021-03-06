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


from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import confusion_matrix

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier


import random
from collections import Counter

import final as fi



# == Parametros
n_neighbors = 20
n_mfcc = 40 # 40
seed = 7

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

param_grid_rf = {
    'n_jobs': [2,3,4,5,6],
    'random_state': [1,2,3]
}

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
    if(i%5 == 0 ): print("Treino",i," de ",tam)
    
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
    "EXTRA TREE",
    "KNN - distance",
    "KNN - uniform",
    "Naive Bayes",
    "SVM",
    "Gaussian Process",
    "Neural Net",
    "Bagging"
]

models = [
    GridSearchCV(RandomForestClassifier(),param_grid_rf),
    #RandomForestClassifier(n_jobs=4, random_state=1),
    #RandomForestClassifier(n_jobs=4, random_state=1, max_depth=None, n_estimators=10),
    ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0),
    neighbors.KNeighborsClassifier(n_neighbors, weights='distance'),
    neighbors.KNeighborsClassifier(n_neighbors, weights='uniform'),
    GaussianNB(),
    GridSearchCV(SVC(kernel='rbf', class_weight='balanced',probability=True), param_grid), #PCA(svd_solver='randomized',whiten=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    MLPClassifier(random_state=1,hidden_layer_sizes=(32,16),activation='relu'),
    BaggingClassifier(n_jobs=3, random_state=0)
    ]


# Selecionando modelos e setando pesos relativos
print("Selecionando modelos")

selected_models = [0,1,4,5,6]
#selected_models = [0,1,2,3,4,5,6]
#selected_models = [0,1,4,5]
relative_wheigths = [2,2,1,4,3,2]

aux1 = []
aux2 = []
estimators = []
for i in selected_models:
    aux1.append(models_name[i])
    aux2.append(models[i])
    estimators.append((models_name[i],models[i]))

models_name = aux1
models = aux2

# Treinando os modelos

print("\n#ETAPA 3 - Treinamento")

# ENSEMBLE

print("Treinando pelo Ensemble")
#eclf1 = VotingClassifier(estimators=estimators, weights=relative_wheigths, voting='soft', n_jobs=3)
eclf1 = VotingClassifier(estimators=estimators, voting='hard', n_jobs=1)
eclf1 = eclf1.fit(X, y)


# Predicao do teste
#'''
print("\n#ETAPA 4 - Teste/Validação")

print("Predizendo - Final")
j = 0
i = 0
captcha_solveds = 0
success = 0

for ytn in yt:
    i = 0
    result = []
 
     #debug
    if(j%10 == 0): print("Preditos:",j,"/",len(yt))
    
    #Predizendo letra a letra
    for k in range(4):
        yyy = eclf1.predict([Xt[j][k]])[0]
        result.append(yyy)
    result = np.array(result)
    #print(result, ytn)
    c = ytn[ytn == result]
    success += len(c)
    if(len(c) == 4): captcha_solveds +=1
    #print(j)
    j+=1

# Printando resultados
print("\n#ETAPA 5 - RESULTADOS: ")
print("Caracteres corretos: ",success,"de",len(yt)*4)
print("Captchas corretos: ",captcha_solveds,"de",len(yt))


#'''

# Código utilizado para selecao dos modelos
# ==== ANALISANDO MODELOS
'''
i = 0
print("Treinando modelos")
for model in models:
    print("\nTreinando: "+models_name[i])
    model.fit(X,y)
    i+=1

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
