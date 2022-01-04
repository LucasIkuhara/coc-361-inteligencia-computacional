# %%
# Imports
import matplotlib.pyplot as plt
from numpy.lib.arraypad import _pad_dispatcher, pad
import pandas as pd
import numpy as np
import os
import plotly.express as px


# %%
# Leitura do dataset

df = pd.read_csv('../marketing_data.csv')
df.head()

# %%
# Tratamento básico e estatísticas

print('Estatistícas do Dataset', '='*50)
print(f'O dataset contém {len(df)} linhas, das quais {len(df[df.duplicated()])} são duplicatas.')
print(f'Cada uma das linhas tem {len(df.columns)} colunas.')

# Limpando valores NaN, Null ou Blank.
df = df.dropna()
print(f'''Após a remoção de linhas faltando dados, ou preenhidas incorretamente (NaN, null..),
ficamos com {len(df)}  linhas.''')

# Remover coluna ID
df = df.drop('ID', axis=1)
print(f'''Além disso, a coluna ID foi removida, pois não agrega a nossa exploração.
Dessa maneira, ficamos com {len(df.columns)} colunas.''')

# %%
# Histograma de cada coluna

base = os.getcwd()  # Salva endereço atual
os.chdir('./histogramas/')  # Muda de diretório

nao_criados = []  # Salvar referências de falhas

# Gerar e salvar gráficos
for column in df.columns:
    try:
        # Caso a coluna tenha poucas entradas únicas
        # Usar a quantidade de entradas como quantidade de bins
        # Caso o contrário, usar 10
        if bins := len(df[column].unique()) > 10:
            bins = 10

        fig = px.histogram(df, x=column, nbins=bins)
        fig.show()
        print(f"Gráfico acima: {column}")
        fig.write_image(f"{column}.png")

    except Exception:
        nao_criados.append(column)

# Listar eventuais falhas
if len(nao_criados) > 0:
    print(f'Gráficos não salvos: ')
    for nome in nao_criados:
        print(f'  -{nome}')

os.chdir(base)  # Volta ao diretório original

# %%
# Correlação entre variáveis

f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar(extend='both')
plt.clim(-1, 1)
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16, pad=80)
plt.savefig('matrix_de_correlacao.png')

# Referência: https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas

# %%
