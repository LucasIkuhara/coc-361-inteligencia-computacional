# %%
# Importando pacotes
import numpy as np
import pandas as pd
from sklearn import tree, ensemble, svm, naive_bayes
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import plotly.express as px

# %%
# Ler dados
df = pd.read_csv('treated_marketing_data.csv')
print('Sample data:')
print(df.tail())

# %%
# Separando Df de features e resultados

training_targets = df['Response']
training_data = df.drop('Response', axis=1)

# %%
# Treinamento com validação cruzada (K-Fold CV)


# Avaliação de modelo com validação cruzada (k=10 por default)
def k_fold_cv(model, k=10):
    score = cross_val_score(model, training_data, training_targets, cv=k)
    print(f'Acuracia: {score.mean():.3f}, Desvio padrão: {score.std():.3f}', end='\n\n')

    return (score.mean(), score.std())


# Testando diferentes modelos
# %%
# Árvores de Decisão
print('Testando árvores de decisão:')

print('Critério "gini", profundidade máxima ilimitada')
model = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
k_fold_cv(model)

print('Critério "entropy", profundidade máxima ilimitada')
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)
k_fold_cv(model)

# %%
# Gradient Boosting
print('Testando gradient boosting:')

print('Função de perda "deviance", quantidade de estimadores 100')
model = ensemble.GradientBoostingClassifier(loss='deviance', n_estimators=100)
k_fold_cv(model)
print('Função de perda "exponential", quantidade de estimadores 100')
model = ensemble.GradientBoostingClassifier(loss='exponential', n_estimators=100)
k_fold_cv(model)

# %%
# SVM
print('Testando SVMs:')

print('Classificador com C=1')
model = svm.SVC(C=1)
k_fold_cv(model)

print('Classificador com C=2')
model = svm.SVC(C=2)
k_fold_cv(model)

# %%
# Random Forest
print('Testando Random Forest:')

print('Critério "gini", quantidade de estimadores 100')
model = ensemble.RandomForestClassifier(criterion='gini', n_estimators=100)
k_fold_cv(model)
print('Critério "entropy", quantidade de estimadores 100')
model = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=100)
k_fold_cv(model)

# %%
# Gaussian Naive Bayes
print('Testando Gaussian Naive Bayes:')

print('Var smoothing = 1e-9')
model = naive_bayes.GaussianNB()
k_fold_cv(model)

print('Var smoothing = 1e-11')
model = naive_bayes.GaussianNB(var_smoothing=float('1e-11'))
k_fold_cv(model)
