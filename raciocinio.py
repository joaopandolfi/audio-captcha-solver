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
    "Neural Net"
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
    ]


# Selecionando modelos e setando pesos relativos
print("Selecionando modelos")

selected_models = [0,1,5,6,4]
#selected_models = [0,1,2,3,4,5,6]
#selected_models = [0,1,4,5]
relative_wheigths = [2,2,4,3,1]

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

'''
lenData = len(y)
X1 = X[int(lenData*0.3):]
y1 = y[int(lenData*0.3):]

X2 = X[:int(lenData*0.3)]
y2 = y[:int(lenData*0.3)]
'''

'''
i = 0
print("Treinando modelos")
for model in models:
    print("\nTreinando: "+models_name[i])
    model.fit(X,y)
    i+=1
'''

# ENSEMBLE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

print("Treinando pelo Ensemble")
eclf1 = VotingClassifier(estimators=estimators, voting='soft', n_jobs=3)
eclf1 = eclf1.fit(X, y)

'''
# Stack de metodos
mStack = GaussianNB()
#mStack = RandomForestClassifier(n_jobs=4, random_state=1)
#mStack = MLPClassifier(random_state=1,hidden_layer_sizes=(32,16),activation='relu')
#mStack = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)

print("Treinando stack")
i = 0
#yy = y[:29]
Xs = []
for yn in y:
    p_temp = []
    #debug
    if(i%10 == 0): print("Retreinados:",i,"/",len(y))
    j = 0
    for model in models:
        #print("\nPredizendo: "+models_name[i])
        p_temp.append(mc_labels.index(model.predict_proba([X[i]])))
        #print(j)
        #for w in range(relative_wheigths[j]): # Peso relativo
        #        p_temp.append(mc_labels.index(model.predict([X[i]])[0])+1)
        j+=1

    #print(np.array([p_temp]),[yn])
    #mStack.fit(np.array([p_temp]),[yn])
    Xs.append(p_temp)
    i+=1

mStack.fit(np.array(Xs),y)
'''

# Predicao do teste
#'''
print("\n#ETAPA 4 - Teste/Validação")

print("Predizendo - Final")
j = 0
i = 0
captcha_solveds = 0
success = 0
statistics = [[],[]] # [pred, labels]


p_model = 2

for ytn in yt:
    i = 0
    predicteds = [[],[],[],[]]
    result = []
 
    p_temp = [[],[],[],[]]
    p_temp_code = [[],[],[],[]]
     #debug
    if(j%10 == 0): print("Preditos:",j,"/",len(yt))
    '''
    for model in models:
        #print("\nPredizendo: "+models_name[i])
        p = model.predict_proba(Xt[j])
        for k in range(4):
            p_temp[k].append(p[k])
            p_temp_code[k].append(mc_labels.index(p[k])+1)

            #for w in range(relative_wheigths[i]): # Peso relativo
            #    predicteds[k].append(mc_labels.index(p[k])+1)
        i+=1
    '''
    
    #print(predicteds)
    #Escolhendo
    for k in range(4):
        '''
        # Stack
        #yyy= mStack.predict(np.array([p_temp_code[k]]))
        #print(p_temp_code[k],yyy)
        #result.append(yyy[0])
        '''
        yyy = eclf1.predict([Xt[j][k]])[0]
        #print(yyy)
        result.append(yyy)

        '''
        # Euristica mista
        if(p_temp[k][p_model] in ['n','b']): # Se for algum que o SVM entra em colizao# ->n,b,6
            if(p_temp[k][p_model] == 'n'): # SVM tem mta colizao com m (diz que e 'm') - se for desempata com outros modelos
                if(p_temp[k][4] == 'h'): # Se NB Disse que e h -> prob de ser N aumenta
                    for w in range(relative_wheigths[4]):
                        predicteds[k].append('m')
                if(p_temp[k][3] == 'n'): # Se Gaussian Disse que e n -> prob de ser M aumenta
                    for w in range(relative_wheigths[3]):
                        predicteds[k].append('m')
            elif(p_temp[k][p_model] == 'b'): # fala que e b mas a maioria das vezes e d
                #Gaussian confunde pouco B com D
                if(p_temp[k][3] == 'b'): # Confio no Gaussian se ele disser que e 'b'
                    for w in range(relative_wheigths[3]):
                        predicteds[k].append('b')
            aux = predicteds[k]#[relative_wheigths[0]:]
            ct = Counter(aux)
            ct = ct.most_common()[0][0]
            result.append(ct)
        else: # se nao, SVM manda
            result.append(p_temp[k][p_model])   
        #ct = Counter(predicteds[k])
        #ct = ct.most_common()[0][0]
        #result.append(ct)
        '''
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
c_models = []
c_models.append(models)
c_models.append(mStack)
j = 0
print("Predizendo")
for model in c_models:
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

# == BAGGING
'''
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

seed = 7

kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
#estimators = []
#model1 = LogisticRegression()
#estimators.append(('logistic', model1))
#model2 = DecisionTreeClassifier()
#estimators.append(('cart', model2))
#model3 = SVC()
#estimators.append(('svm', model3))
# create the ensemble model


ensemble = VotingClassifier(models)
results = model_selection.cross_val_score(ensemble, X, y, cv=kfold)
print(results.mean())



kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#print("\n\nTREINO ",y)
#print("\nTESTE: ",yt)

'''