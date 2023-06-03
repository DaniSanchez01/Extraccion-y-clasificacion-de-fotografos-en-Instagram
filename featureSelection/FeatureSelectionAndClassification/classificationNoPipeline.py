import json
import math
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from tools.data_processing import FeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier




dataset = pd.read_csv('datasetFeatureSelected.csv', header=0)

# split into input (X) and output (y) variables
input = dataset.iloc[:, :-1]

'''input.drop('isProfessional', axis=1, inplace=True)
input.drop('avgTaggedUsers', axis=1, inplace=True)
input.drop('varTaggedUsers', axis=1, inplace=True)'''

col = input.columns

output = dataset.values[:,-1]
output=output.astype('int')

#Escalabilización de los datos
# Inicializar StandardScaler
scaler = StandardScaler()

# Ajustar StandardScaler a los datos
scaler.fit(input)

# Escalar los datos
inputScaled = scaler.transform(input)

inputScaled = pd.DataFrame(input, columns=col)

#x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.33, random_state=1)


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
for model in models:
    usedInput = inputScaled
    if (not os.path.exists(f"./param{name_model[i]}.json")):
        grid_search = GridSearchCV(model, scoring='roc_auc', n_jobs= -1, verbose = 1, param_grid=grids[i], cv=5)
        grid_search.fit(usedInput, output)
        best_params = grid_search.best_params_

        with open(f"./param{name_model[i]}.json", 'w') as f:
            json.dump(best_params, f)
    else:
        with open(f"./param{name_model[i]}.json", 'r') as f:
            best_params = json.load(f)

    model.set_params(**best_params)    
    y_pred = cross_val_predict(model, usedInput, output, cv=5)
    y_pred_scores = cross_val_predict(model, usedInput, output, cv=5, method="predict_proba")
    y_pred_proba = y_pred_scores[:,-1]

    print(name_model[i])
    print()
    print(metrics.confusion_matrix(output, y_pred))
    print(metrics.classification_report(output,y_pred))

    # Calcular tasas de verdaderos y falsos positivos para diferentes umbrales
    fpr, tpr, thresholds = metrics.roc_curve(output, y_pred_proba)
    roc_auc = metrics.auc(fpr, tpr)
    print("AUC_score: ",round(roc_auc,4))
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
    
    i+=1
