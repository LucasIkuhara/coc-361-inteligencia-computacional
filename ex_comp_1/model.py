# %%
# Importando pacotes
import numpy as np
import pandas as pd
from sklearn import tree
import plotly.express as px

# %%
# Ler dados
df = pd.read_csv('../star-dataset/Star3642_balanced.csv')
print('Sample data:')
print(df.tail())

# %%
# Visualização inicial

# Função de plot
def plot(subtitulo=False):

    # Configuração dos gráficos
    config = {
        'displaylogo': True,
        'logo': False
    }

    for coluna in df.columns:
        titulo = coluna
        if subtitulo: titulo += f': {subtitulo}'
        vis = px.histogram(df, x=coluna, color='TargetClass', template='plotly_dark', title=titulo)
        vis.show(config=config)


plot()

# %%
# Limpeza e tratamento de dados

apicar_log = ()

for coluna in apicar_log:
    df[coluna] = (df[coluna]).transform(np.log)
fig = px.histogram(df2, x='SpType', y='occ_percetual')
fig.show()
# print(df2.tail())

# #%%
# # Separando Df de features e resultados
# training_targets = df['TargetClass']
# training_data = df.drop('TargetClass', axis=1)
# training_data = training_data.drop('SpType', axis=1)  # TODO Mapear os tipos para tipo int

# # Treinar árvore de decisão
# decision_tree = tree.DecisionTreeClassifier()
# decision_tree.fit(training_data, training_targets)a


# tree.plot_tree(decision_tree)

# %%
