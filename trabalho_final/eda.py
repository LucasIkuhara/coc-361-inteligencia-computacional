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
# Histograma de cada coluna

base = os.getcwd()  # Salva endereço atual
os.chdir('./histogramas/')  # Muda de diretório

nao_criados = []  # Salvar referências de falhas

# Gerar e salvar gráficos
for column in df.columns:
    try:
        fig = px.histogram(df, x=column, nbins=10)
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
plt.title('Correlation Matrix', fontsize=16, pad=100)

# Referência: https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas

# %%
