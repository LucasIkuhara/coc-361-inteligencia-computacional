# %%
# Imports
from model import Classificador
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from random import shuffle


# %%
# Definição de funções auxiliares
def dividir_dataset(razao_treino: float, dataset):
    '''dividir_dataset(razao_treino, dataset) -> dataset1, dataset2.
    Divide o dataset de acordo com a razão escolhida. '''
    if 0 > razao_treino > 1:
        raise Exception('Razão fora do intervalo (0, 1).')

    tamanho = int(len(dataset) * razao_treino)
    return dataset[0:tamanho], dataset[tamanho:]


# %%
# Dataset
df = pd.read_csv('../pokemon_dataset/pokemon.csv')

# Excluindo tipos secundários
df.drop('Type2', axis=1)

# Mapear todos os tipos
tipos = df.Type1.unique()
print("Tipos encontrados: ")
for tipo in tipos:
    print(f'  -{tipo}')

# Ler nomes das imagens
os.chdir('../pokemon_dataset/images/images/')
files = os.listdir()
shuffle(files)  # Emnaralhar dataset

# Ler imagens
x = [plt.imread(f'./{name}', )
     [:, :, :3]  # Descartar canal alpha
     for name in files
     ]

# Ler labels
y = [df[df.Name == name.split('.')[0]].Type1 for name in files]
y = np.array(y).flatten()

x_treino, x_validacao = dividir_dataset(0.5, x)
y_treino, y_validacao = dividir_dataset(0.5, y)

# %%
# Treinamento
# Baseado em exemplos de
# https://www.tensorflow.org/tutorials/images/cnn?hl=pt-br
classificador = Classificador()

classificador.model.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                from_logits=True
                            ),
                            metrics=['accuracy'])

history = classificador.model.fit(x_treino,
                                  y_treino,
                                  epochs=10,
                                  batch_size=len(x_treino),
                                  validation_data=(x_validacao, y_validacao)
                                  )

# %%
