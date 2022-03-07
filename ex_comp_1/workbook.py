# %%
# Importando pacotes
import numpy as np
import pandas as pd
from sklearn import tree, ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import plotly.express as px

# %%
# Ler dados
df = pd.read_csv('./star-dataset/Star3642_balanced.csv')
print('Sample data:')
print(df.tail())

# %%
# Visualização inicial

# Configuração dos gráficos
config = {
    'displaylogo': False,
    'scrollZoom': True
}


# Função de plot
def plot(subtitulo=False):

    # Gerar um histograma por coluna
    for coluna in df.columns:
        titulo = coluna
        if subtitulo:
            titulo += f': {subtitulo}'

        vis = px.histogram(df, x=coluna, color='TargetClass',
                           template='plotly_dark', title=titulo,
                           marginal="violin")
        vis.show(config=config)


plot()

# %%
# Correlação entre variáveis

fig = px.imshow(df.corr(), template='plotly_dark',
                text_auto=True, title='Matiz de Correlação',
                color_continuous_scale=px.colors.sequential.Agsunset,
                range_color=(-1, 1))

fig.show(config=config)

# %%
# Limpeza e tratamento de dados
# Referência: https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114

# Retirar nulos, NaN e NA
df = df.dropna()
print(f'''Após a remoção de linhas faltando dados, ou preenhidas incorretamente (NaN, null..),
ficamos com {len(df)}  linhas.''')

# Dropar a coluna SpType (584 categorias diferentes, menos de 1% de ocorrência
# da classe mais representada)
df = df.drop('SpType', axis=1)

# Transformações de Log
colunas_alvo_log = ('Plx', 'e_Plx')

for coluna in colunas_alvo_log:
    df[coluna] = (df[coluna] + 1).transform(np.log)


# Standartização
def standardize(coluna):
    return (coluna - coluna.mean()) / coluna.std()


colunas_alvo_std = ('Vmag', 'Plx', 'e_Plx', 'B-V', 'Amag')
for coluna in colunas_alvo_std:
    df[coluna] = (df[coluna]).transform(standardize)

# Retirar nulos, NaN e NA que podem ser gerados durante os ajustes
df = df.dropna()

plot('ajustado')

# %%
# Separando Df de features e resultados

training_targets = df['TargetClass']
training_data = df.drop('TargetClass', axis=1)

# %%
# Treinamento com validação cruzada (K-Fold CV)


# Avaliação de modelo com validação cruzada (k=10 por default)
def k_fold_cv(model, k=10):
    score = cross_val_score(model, training_data, training_targets, cv=k)
    print(f'Acuracia: {score.mean():.3f}, Desvio padrão: {score.std():.3f}', end='\n\n')

    return (score.mean(), score.std())


# Testando diferentes modelos
print('Testando árvores de decisão:')

print('Critério "gini", profundidade máxima 10')
model = tree.DecisionTreeClassifier(criterion='gini', max_depth=10)
k_fold_cv(model)

print('Critério "gini", profundidade máxima ilimitada')
model = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
k_fold_cv(model)

print('Critério "entropy", profundidade máxima 10')
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
k_fold_cv(model)

print('Critério "entropy", profundidade máxima ilimitada')
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)
k_fold_cv(model)

print('Testando gradient boosting:')

print('Função de perda "deviance", quantidade de estimadores 100')
model = ensemble.GradientBoostingClassifier(loss='deviance', n_estimators=100)
k_fold_cv(model)

print('Função de perda "deviance", quantidade de estimadores 200')
model = ensemble.GradientBoostingClassifier(loss='deviance', n_estimators=200)
k_fold_cv(model)

print('Função de perda "exponential", quantidade de estimadores 100')
model = ensemble.GradientBoostingClassifier(loss='exponential', n_estimators=100)
k_fold_cv(model)

print('Função de perda "exponential", quantidade de estimadores 200')
model = ensemble.GradientBoostingClassifier(loss='exponential', n_estimators=200)
k_fold_cv(model)
