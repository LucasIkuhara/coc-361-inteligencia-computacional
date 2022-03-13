# %%
# Importando pacotes
import numpy as np
import pandas as pd
from sklearn import tree, ensemble, svm, naive_bayes
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from utils import plot_confusion, additional_metrics
import pandas as pd
import numpy as np

# %%
# Ler dados
df = pd.read_csv('treated_marketing_data.csv')
print('Sample data:')
print(df.tail())

# %%
# Separando Df de features e resultados

training_targets = df['Response'].to_numpy()
training_data = df.drop('Response', axis=1).to_numpy()

# %%
# Função de treinamento com validação cruzada (K-Fold CV)


# Referência:
# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
# Avaliação de modelo com validação cruzada (k=10 por default)
def k_fold_cv(model, k=10):
    '''
    # k_fold_cv
    Avalia modelos com interfaces padrão Sklearn.

    Args:
        model - Modelo Sklearn ou compatível.

        k - quantidade de divisões do dataset para validação cruzada.

    Returns:
        (acurácia média, desvio padrão, predições, respostas)
    '''

    k_fold = KFold(10)
    predictions = np.array([])
    prediction_targets = np.array([])
    scores = np.array([])

    for train, test in k_fold.split(training_data, training_targets):

        model.fit(training_data[train], training_targets[train])
        score = model.score(training_data[test], training_targets[test])

        scores = np.append(scores, score)
        predictions = np.append(predictions, model.predict(training_data[test]))
        prediction_targets = np.append(prediction_targets, training_targets[test])

    print(f'Acurácia: {scores.mean():.3f}, Desvio padrão: {scores.std():.3f}', end='\n\n')

    return(scores.mean(), scores.std(), predictions, prediction_targets)


# Testando diferentes modelos
# %%
# Árvores de Decisão
print('Testando árvores de decisão:')

print('Critério "gini", profundidade máxima ilimitada')
model = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
result = k_fold_cv(model)
conf_matrix = confusion_matrix(result[3], result[2])
additional_metrics(conf_matrix)
plot_confusion(conf_matrix, 'Árvore de Decisão (gini)')


print('Critério "entropy", profundidade máxima ilimitada')
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)
result = k_fold_cv(model)
conf_matrix = confusion_matrix(result[3], result[2])
additional_metrics(conf_matrix)
plot_confusion(conf_matrix, 'Árvore de Decisão (entropy)')


# %%
# Gradient Boosting
print('Testando gradient boosting:')

print('Função de perda "deviance", quantidade de estimadores 100')
model = ensemble.GradientBoostingClassifier(loss='deviance', n_estimators=100)
result = k_fold_cv(model)
conf_matrix = confusion_matrix(result[3], result[2])
additional_metrics(conf_matrix)
plot_confusion(conf_matrix, 'Gradient Boosting (deviance)')

print('Função de perda "exponential", quantidade de estimadores 100')
model = ensemble.GradientBoostingClassifier(loss='exponential', n_estimators=100)
result = k_fold_cv(model)
conf_matrix = confusion_matrix(result[3], result[2])
additional_metrics(conf_matrix)
plot_confusion(conf_matrix, 'Gradient Boosting (exponential)')


# %%
# SVM
print('Testando SVMs:')

print('Classificador com C=1')
model = svm.SVC(C=1)
result = k_fold_cv(model)
conf_matrix = confusion_matrix(result[3], result[2])
additional_metrics(conf_matrix)
plot_confusion(conf_matrix, 'Support Vector Machine (C=1)')


print('Classificador com C=2')
model = svm.SVC(C=2)
result = k_fold_cv(model)
conf_matrix = confusion_matrix(result[3], result[2])
additional_metrics(conf_matrix)
plot_confusion(conf_matrix, 'Support Vector Machine (C=2)')


# %%
# Random Forest
print('Testando Random Forest:')

print('Critério "gini", quantidade de estimadores 100')
model = ensemble.RandomForestClassifier(criterion='gini', n_estimators=100)
result = k_fold_cv(model)
conf_matrix = confusion_matrix(result[3], result[2])
additional_metrics(conf_matrix)
plot_confusion(conf_matrix, 'Random Forest (gini)')

print('Critério "entropy", quantidade de estimadores 100')
model = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=100)
result = k_fold_cv(model)
conf_matrix = confusion_matrix(result[3], result[2])
additional_metrics(conf_matrix)
plot_confusion(conf_matrix, 'Random Forest (entropy)')


# %%
# Gaussian Naive Bayes
print('Testando Gaussian Naive Bayes:')

print('Var smoothing = 1e-9')
model = naive_bayes.GaussianNB()
result = k_fold_cv(model)
conf_matrix = confusion_matrix(result[3], result[2])
additional_metrics(conf_matrix)
plot_confusion(conf_matrix, 'Gaussian Naive Bayes (1e-9)')


print('Var smoothing = 1e-11')
model = naive_bayes.GaussianNB(var_smoothing=float('1e-11'))
result = k_fold_cv(model)
conf_matrix = confusion_matrix(result[3], result[2])
additional_metrics(conf_matrix)
plot_confusion(conf_matrix, 'Gaussian Naive Bayes (1e-11)')


# %%
# Logistic Regression
print('Testando Regressão Logística:')

print('Solver lbfgs')
model = LogisticRegression(solver='lbfgs', max_iter=5000)
result = k_fold_cv(model)
conf_matrix = confusion_matrix(result[3], result[2])
additional_metrics(conf_matrix)
plot_confusion(conf_matrix, 'Regressão Logística (lbfgs)')


print('Solver SAGA')
model = LogisticRegression(solver='sag', max_iter=5000)
result = k_fold_cv(model)
conf_matrix = confusion_matrix(result[3], result[2])
additional_metrics(conf_matrix)
plot_confusion(conf_matrix, 'Regressão Logística (saga)')
