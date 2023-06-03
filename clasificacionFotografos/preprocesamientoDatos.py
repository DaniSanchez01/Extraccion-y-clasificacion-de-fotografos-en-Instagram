import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from tools.data_processing import FeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from joblib import dump, load



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

    tamanos = [fotografos, normales]

    etiquetas = [f'Fotógrafos profesionales - {tamanos[1]}', f'Otros usuarios - {tamanos[0]}']


    plt.pie(tamanos, labels=etiquetas, autopct = '%1.1f%%', explode = [0, 0.1])

    plt.show()

# Lee el archivo CSV y lo convierte en un DataFrame
dataset = pd.read_csv('dataset.csv', header=0)
plt.rcParams.update({'font.size': 12}) 

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

show_loan_distrib(output)

###############################################################################
#                            Train/test division                              #
###############################################################################
#Crear los conjuntos de entrenamiento y test
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.30, random_state=1)

###############################################################################
#                            Escalamos los datos                              #
###############################################################################
# Inicializar StandardScaler
scaler = StandardScaler()

# Ajustar StandardScaler a los datos
scaler.fit(x_train)

# Escalar los datos
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


x_train_scaled = pd.DataFrame(x_train_scaled, columns=col)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=col)

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
fs.fit(x_train_scaled, y_train, steps)

# Transformar el conjunto de datos, eliminando las características calculadas como constantes o correlacionadas
X_selected_train = fs.transform(x_train_scaled)
X_selected_test = fs.transform(x_test_scaled)

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

###############################################################################
#                       Prediccion sin Feature Selection                      #
###############################################################################
print("------------------------")
print("Predict without Feature Selection")
print("------------------------")

estimator = LogisticRegression()

# Define parameter grid
param_grid = { 
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear']
    }

#Si no tenemos guardados los mejores parámetros para un random forest, calcularlos
if (not os.path.exists(f"./nonFsLogisticRegression.joblib")):

    gscv = GridSearchCV(estimator, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = 'roc_auc')
    gscv.fit(x_train_scaled, y_train)

    # Guarda los mejores parámetros para el modelo en un csv
    best_estimator = gscv.best_estimator_
    dump(best_estimator, 'nonFsLogisticRegression.joblib')


#Si ya habiamos calculado el mejor modelo, cogerlo de un archivo joblib
else:
    best_estimator = load('nonFsLogisticRegression.joblib')

# Predecir
t1 = time.time()
y_pred_test = best_estimator.predict(x_test_scaled)
t2 = time.time()
print((t2-t1)*1000," milisegundos")

#Mostrar resultados (Matriz de confusión y tabla de resultados)
print(metrics.confusion_matrix(y_test,y_pred_test))
print(metrics.classification_report(y_test,y_pred_test))
y_pred_test_proba = best_estimator.predict_proba(x_test_scaled)[:,-1]

#Calcular la curva ROC y su área (AOC)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_test_proba)
roc_auc = metrics.auc(fpr, tpr)
print("AUC_score: ", round(roc_auc,4))
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
estimator2 = LogisticRegression(random_state=42)

#Si no tenemos guardados los mejores parámetros para un random forest, calcularlos
if (not os.path.exists(f"./fsLogisticRegression.joblib")):

    gscv = GridSearchCV(estimator2, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = 'roc_auc')
    gscv.fit(X_selected_train, y_train)

    # Guarda los mejores parámetros para el modelo en un csv
    best_estimator = gscv.best_estimator_
    dump(best_estimator, 'fsLogisticRegression.joblib')


#Si ya habiamos calculado el mejor modelo, cogerlo de un archivo joblib
else:
    best_estimator = load('fsLogisticRegression.joblib')

# Predecir
t3 = time.time()
y_pred_test = best_estimator.predict(X_selected_test)
t4 = time.time()
print((t4-t3)*1000," milisegundos")

#Mostrar resultados (Matriz de confusión y tabla de resultados)
print(metrics.confusion_matrix(y_test,y_pred_test))
print(metrics.classification_report(y_test,y_pred_test))
y_pred_test_proba = best_estimator.predict_proba(X_selected_test)[:,-1]

#Calcular la curva ROC y su área (AOC)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_test_proba)
roc_auc = metrics.auc(fpr, tpr)
print("AUC_score: ", round(roc_auc,4))
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
