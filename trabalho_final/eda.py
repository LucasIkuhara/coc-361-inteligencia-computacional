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
df = pd.read_csv('marketing_data.csv')
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

        vis = px.histogram(df, x=coluna, color='Response',
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

# Dropar coluna de ID
df = df.drop('ID', axis=1)

# Retirar espaço do nome da coluna Income e transformar em valor numérico
df["Income"] = df[" Income "].replace("[$, ]", "", regex=True).astype(float)
df = df.drop("Income ", axis=1)

# Remoção de outliers encontrados
df = df.drop(df[df['Year_Birth'] < 1900].index)  # Datas de nascimento antes de 1920
df = df[df['Marital_Status'].isin(('Divorced', 'Single', 'Married', 'Together', 'Widow'))]

# Transformar datas em números

# Transformações de Log
colunas_alvo_log = (
    'MntWines',
    'MntFruits',
    'MntMeatProducts',
    'MntFishProducts',
    'MntSweetProducts',
    'MntGoldProds'
    )

for coluna in colunas_alvo_log:
    df[coluna] = (df[coluna] + 1).transform(np.log)


# One-hot encode
def encode_as_one_hot(df, nome_da_coluna):
    # Referência: https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114#199b
    encoded_columns = pd.get_dummies(df[nome_da_coluna])
    df = df.join(encoded_columns).drop('column', axis=1)

    return df


colunas_alvo_ohe = (
    ''
)

# Standartização
def standardize(coluna):
    return (coluna - coluna.mean()) / coluna.std()


colunas_alvo_std = df.columns
for coluna in colunas_alvo_std:
    df[coluna] = (df[coluna]).transform(standardize)

# Retirar nulos, NaN e NA que podem ser gerados durante os ajustes
df = df.dropna()

# %%
# Salvar valores tratados para treino
df.to_csv('treated_marketing_data.csv')
