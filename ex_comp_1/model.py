# %%
# Importando pacotes
import numpy as np
import pandas as pd
from sklearn import tree, ensemble
import matplotlib.pyplot as plt
from numpy.lib.arraypad import _pad_dispatcher, pad
import pandas as pd
import numpy as np
import os
import plotly.express as px

# %%
# Ler dados
df = pd.read_csv('../star-dataset/Star3642_balanced.csv')
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
    df[coluna] = df[coluna].transform(np.log)


# Standartização
def standardize(coluna):
    return (coluna - coluna.mean()) / coluna.std()


colunas_alvo_std = ('Vmag', 'Plx', 'e_Plx', 'B-V', 'Amag')
for coluna in colunas_alvo_std:
    df[coluna] = (df[coluna]).transform(standardize)

plot('ajustado')

# %%
# print(df2.tail())

# #%%
# # Separando Df de features e resultados
# training_targets = df['TargetClass']
# training_data = df.drop('TargetClass', axis=1)
# training_data = training_data.drop('SpType', axis=1)  # TODO Mapear os tipos para tipo int
# Tratamento básico


# %%
# Separando Df de features e resultados
training_targets = df['TargetClass']
training_data = df.drop('TargetClass', axis=1)

# # Treinar árvore de decisão
# decision_tree = tree.DecisionTreeClassifier()
# decision_tree.fit(training_data, training_targets)a


# tree.plot_tree(decision_tree)

# %%
# Treinar
grad_boosting = ensemble.GradientBoostingClassifier()

# tree.plot_tree(decision_tree)
