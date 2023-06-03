import json
import os
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load

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


    print("Perfiles normales = ",normales)
    print("Perfiles fotografos = ",fotografos)

    plt.pie(tamanos, labels=etiquetas, autopct = '%1.1f%%', shadow = True, explode = [0, 0.1])

    plt.show()


trainDataset = pd.read_csv('trainDataset.csv', header=0)

# split into input (X) and output (y) variables
x_train = trainDataset.iloc[:, :-1]
y_train = trainDataset.values[:,-1]

testDataset = pd.read_csv('testDataset.csv', header=0)

# split into input (X) and output (y) variables
x_test = testDataset.iloc[:, :-1]
y_test = testDataset.values[:,-1]

# definir modelos de clasificación
models = [LogisticRegression(), GradientBoostingClassifier(), DecisionTreeClassifier(), RandomForestClassifier(random_state=42)]
#, SVC(), RandomForestClassifier(n_estimators=100, random_state=42
name_model = ["LogisticRegression", "GradientBoosting", "DecisionTrees", "RandomForest"]
grids = [
    {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear']
    },
    {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.5],
        'max_depth': [3, 6],
        'min_samples_split': [2, 4]

    },
    {
        'criterion': ['entropy', 'gini'],
        'min_samples_leaf': [1, 2, 3],
        'max_depth': [1, 2, 3]
    },
    { 'n_estimators': [200],
                'class_weight': [None, 'balanced'],
                'max_features': ['sqrt', 'log2'],
                'max_depth' : [6, 7, 8],
                'min_samples_split': [0.005, 0.01, 0.05],
                'min_samples_leaf': [0.005, 0.01, 0.05],
                'criterion' :['gini', 'entropy']     ,
                'n_jobs': [-1]
    }]

# evaluar desempeño de los modelos con cross_val_score
i=0
accuracyList = []
precisionList = []
recallList = []
f1List = []
aucList = []
classes = ["No fotógrafo","Fotógrafo"]

for model in models:
    if (not os.path.exists(f"./best{name_model[i]}.joblib")):
        grid_search = GridSearchCV(model, scoring='roc_auc', n_jobs= -1, verbose = 1, param_grid=grids[i], cv=5)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        #print(grid_search.cv_results_)
        # Obtener los resultados de validación cruzada del mejor modelo
        cv_results = grid_search.cv_results_

        dump(best_model, f'best{name_model[i]}.joblib')
        with open(f"./param{name_model[i]}.json", 'w') as f:
            json.dump(best_params, f)

    else:
        best_model = load(f'best{name_model[i]}.joblib')

    y_pred = best_model.predict(x_test)
    y_pred_proba = best_model.predict_proba(x_test)[:,-1]

    print(name_model[i])
    print()
    cm =metrics.confusion_matrix(y_test, y_pred)
    #print(cm)
    #print(metrics.classification_report(y_train,y_pred))

    # Calcular tasas de verdaderos y falsos positivos para diferentes umbrales
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
    roc_auc = metrics.auc(fpr, tpr)
    print("AUC_score: ",round(roc_auc,4))
    aucList.append(round(roc_auc,3))
    # trazar la curva ROC
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.title(f'ROC for {name_model[i]}',fontsize=25)
    plt.legend(loc="lower right")
    plt.show()
    print()

    # Calcular las métricas de rendimiento
    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracyList.append(round(accuracy,3))
    precision = metrics.precision_score(y_test, y_pred)
    precisionList.append(round(precision,3))
    recall = metrics.recall_score(y_test, y_pred)
    recallList.append(round(recall,3))
    f1 = metrics.f1_score(y_test, y_pred)
    f1List.append(round(f1,3))

    if name_model[i]=="RandomForest":
        #cv_results = cross_validate(model, usedInput, output, cv=5, return_estimator=True)
        importances = best_model.feature_importances_

        # Obtener el índice de las características en orden descendente de importancia
        indices = np.argsort(importances)[::-1]

        # Crear la gráfica de barras
        plt.figure()
        plt.title("Ranking de importancia de características")
        plt.bar(range(x_train.shape[1]), importances[indices])
        plt.xticks(range(x_train.shape[1]), x_train.columns[indices], rotation=90)
        
        # Agregar etiquetas y título al gráfico
        plt.xlabel('Características')
        plt.ylabel('Importancia')
        plt.title('Importancia de las características en Random Forest')
        plt.show()
        
        # Crear la figura y los ejes
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Agregar los números de la matriz en las celdas correspondientes
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="white" if cm[i][j] > cm.max() / 2 else "black")

        # Configurar los ejes
        tick_marks = np.arange(len(classes))
        ax.set(xticks=tick_marks, yticks=tick_marks, xticklabels=classes, yticklabels=classes, xlabel='Predicted label', ylabel='True label')

        # Ajustar el espacio entre las celdas
        plt.tight_layout()

        # Mostrar la gráfica
        plt.show()
    i+=1

i=0
names = ["Accuracy","Precision","Recall","f1","AUC"]
valuesList = [accuracyList, precisionList, recallList, f1List, aucList]
color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

while i<len(valuesList):
    name = names[i]
    measure = valuesList[i]
    # Crear una figura y un eje
    fig, ax = plt.subplots()

    index = np.argsort(measure)[::-1]
    measure_sorted = [measure[i] for i in index]
    model_sorted = [name_model[i] for i in index]
    color_sorted = [color[i] for i in index]

    # Crear la gráfica de barras
    bars = ax.bar(model_sorted, measure_sorted, color=color_sorted, width = 0.65)

    # Agregar etiquetas de valores a las barras
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), 
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',fontsize=16)

    # Agregar título y etiquetas de ejes
    ax.set_title(f'{name} of the models', fontweight='bold', fontsize =20)
    ax.set_xlabel('Modelos',fontsize = 12)
    ax.set_ylabel(name,fontsize = 12)
    plt.yticks(fontsize=14)

    plt.show()
    i+=1