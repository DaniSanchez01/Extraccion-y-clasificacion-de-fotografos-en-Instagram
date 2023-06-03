import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import cross_val_predict
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from tools.data_processing import FeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.svm import SVC



import matplotlib.pyplot as plt


# Crea un gráfico circular con el porcentaje de fotógrafos prefesionales y los que no
def show_loan_distrib(y):
    i=0
    normales = 0
    fotografos = 0
    while (i<len(y)):
        if y[i]==0:
            normales+=1
        else: fotografos+=1
        i+=1

    etiquetas = ['Usuarios normales', 'Fotógrafos profesionales']

    tamanos = [normales, fotografos]


    plt.pie(tamanos, labels=etiquetas, autopct = '%1.1f%%', shadow = True, explode = [0, 0.1])

    plt.show()

# Lee el archivo CSV y lo convierte en un DataFrame
dataset = pd.read_csv('datasetAmpliado.csv', header=0)

# Cogemos todas las características, eliminando la fila de la etiqueta
input = dataset.iloc[:, :-1]

#Quitamos la columna de la característica isProfessional, porque determinamos que estaba muy relacionado
#con la forma en la que etiquetamos a los usuarios
input.drop('isProfessional', axis=1, inplace=True)

#Cogemos el nombre de todas las características
col = input.columns

#Cogemos la columna de las etiquetas y las traducimos a 0s (False) y 1s (True)
output = dataset.values[:,-1]
output=output.astype('int')

#show_loan_distrib(output)

###############################################################################
#                            Escalamos los datos                              #
###############################################################################
# Inicializar StandardScaler
scaler = StandardScaler()

# Ajustar StandardScaler a los datos
scaler.fit(input)

# Escalar los datos
input = scaler.transform(input)

input = pd.DataFrame(input, columns=col)

###############################################################################
#                            Train/test division                              #
###############################################################################
#Crear los conjuntos de entrenamiento y test
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.30, random_state=1)


###############################################################################
#                          Eliminar características                           #
###############################################################################

# Pasos de eliminación
# 1. Eliminar características constantes
step1 = {'Constant Features': {'frac_constant_values': 0.9}}

# 2. Eliminar características correlacionadas
step2 = {'Correlated Features': {'correlation_threshold': 0.9}}

steps = [step1, step2]

print("Iniciar")
# Iniciar FeatureSelector()
fs = FeatureSelector()

# Aplicar los métodos de feature selectionen el orden en el que se ha indicado
fs.fit(x_train, y_train, steps)

# Transformar el conjunto de datos, eliminando las características calculadas como constantes o correlacionadas
X_selected_train = fs.transform(x_train)
X_selected_test = fs.transform(x_test)

#Eliminar características poco importantes
lasso = Lasso(alpha=0.001)

#Entrenar a lasso
lasso.fit(X_selected_train,y_train)
col = X_selected_train.columns

#Lista con los coeficiente de importancia que les ha dado a cada característica
coef = lasso.coef_
selected_features = list()
#Para cada una de ellas
for i in range(len(coef)):
    #Si el coeficiente no es 0, guardar la característica
    if coef[i]!=0:
        selected_features.append(col[i])
    #Si es 0, no guardarla
    else: print(col[i], ':', coef[i])
print(selected_features)
#Transformar los datos, eliminando las características poco importantes
X_selected_train = X_selected_train[selected_features]
X_selected_test = X_selected_test[selected_features]
input = input[selected_features]

#Guardar el dataset, habiendo seleccionado ya las características útiles
selected_features.append("Photographer")
dataset = dataset[selected_features]
print(selected_features)
dataset.to_csv('datasetFeatureSelected.csv', index=False)

###############################################################################
#                       Prediccion sin Feature Selection                      #
###############################################################################
print("------------------------")
print("Predict without Feature Selection")
print("------------------------")

estimator = RandomForestClassifier(random_state=42)

# Define parameter grid
param_grid = { 'n_estimators': [200],
                'class_weight': [None, 'balanced'],
                'max_features': ['sqrt', 'log2'],
                'max_depth' : [6, 7, 8],
                'min_samples_split': [0.005, 0.01, 0.05],
                'min_samples_leaf': [0.005, 0.01, 0.05],
                'criterion' :['gini', 'entropy']     ,
                'n_jobs': [-1]
    }

#Si no tenemos guardados los mejores parámetros para un random forest, calcularlos
if (not os.path.exists(f"./fsRandomForest.json")):

    gscv = GridSearchCV(estimator, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = 'roc_auc')
    gscv.fit(input, output)

    # Guarda los mejores parámetros para el modelo en un csv
    best_params = gscv.best_params_
    with open('fsRandomForest.json', 'w') as f:
        json.dump(best_params, f)

#Si ya habiamos calculado los mejores parámetros, cogerlos de un archivo JSON
else:
    with open('fsRandomForest.json', 'r') as f:
        best_params = json.load(f)

print("Mejores parámetros para Random Forest: ",best_params)
# Establecer los mejores parámetros al modelo
estimator.set_params(**best_params)

# Entrenar al random Forest
estimator.fit(x_train, y_train)

# Predecir
y_pred_test = estimator.predict(x_test)

#Mostrar resultados (Matriz de confusión y tabla de resultados)
print(metrics.confusion_matrix(y_test,y_pred_test))
print(metrics.classification_report(y_test,y_pred_test))
y_pred_test_proba = estimator.predict_proba(x_test)[:,-1]

#Calcular la curva ROC y su área (AOC)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_test_proba)
roc_auc = metrics.auc(fpr, tpr)
print("AUC_score: ", roc_auc)
# trazar la curva ROC
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print()

###############################################################################
#                        Prediccion con Feature Selection                     #
###############################################################################
print("------------------------")
print("Predict with Feature Selection")
print("------------------------")
# Establecer los mejores parámetros al modelo
estimator.set_params(**best_params)

# Entrenar al random Forest
estimator.fit(X_selected_train, y_train)

# Predecir
y_pred_test = estimator.predict(X_selected_test)

#Mostrar resultados (Matriz de confusión y tabla de resultados)
print(metrics.confusion_matrix(y_test,y_pred_test))
print(metrics.classification_report(y_test,y_pred_test))
y_pred_test_proba = estimator.predict_proba(X_selected_test)[:,-1]

#Calcular la curva ROC y su área (AOC)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_test_proba)
roc_auc = metrics.auc(fpr, tpr)
print("AUC_score: ", roc_auc)
# trazar la curva ROC
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print()
