# %%
# Imports
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import cross_val_score, KFold
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
# Treinamento com validação cruzada (K-Fold CV)

# Avaliação de modelo com validação cruzada (k=10 por default)
def k_fold_cv(model, k=10):
    score = cross_val_score(model, training_data, training_targets, cv=k)
    print(f'Acuracia: {score.mean():.3f}, Desvio padrão: {score.std():.3f}', end='\n\n')

    return (score.mean(), score.std())


# %%
# Modelos

class ModelFactory:

    @staticmethod
    def large() -> models.Sequential:

        large_model = models.Sequential()
        large_model.add(layers.Dense(41, activation='relu'))
        large_model.add(layers.Dense(41, activation='relu'))
        large_model.add(layers.Dense(41, activation='relu'))
        large_model.add(layers.Dense(1, activation='sigmoid'))

        large_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return large_model

    @staticmethod
    def medium() -> models.Sequential:

        medium_model = models.Sequential()
        medium_model.add(layers.Dense(41, activation='relu'))
        medium_model.add(layers.Dense(41, activation='relu'))
        medium_model.add(layers.Dense(1, activation='sigmoid'))

        medium_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return medium_model

    @staticmethod
    def small() -> models.Sequential:

        small_model = models.Sequential()
        small_model.add(layers.Dense(41, activation='relu'))
        small_model.add(layers.Dense(1, activation='sigmoid'))

        small_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return small_model


factory = ModelFactory()

# %%
# Treinamento em validação cruzada


# Referência:
# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
def train(model_factory_method):

    kfold = KFold(10, shuffle=False)
    accuracy_tacker = np.array([])
    loss_tracker = np.array([])
    predictions = np.array([])
    prediction_targets = np.array([])

    # Divisão em folds
    for train, test in kfold.split(training_data, training_targets):

        model = model_factory_method()

        model.fit(training_data[train],
                  training_targets[train],
                  batch_size=15,
                  epochs=10,
                  verbose=0)

        # Métricas
        scores = model.evaluate(training_data[test],
                                training_targets[test],
                                verbose=0)
        accuracy_tacker = np.append(accuracy_tacker, scores[1])
        loss_tracker = np.append(loss_tracker, scores[0])

        print(scores[1])

        # Necessários para matriz de confusão
        prediction_targets = np.append(prediction_targets, training_targets[test])
        predictions = np.append(predictions, model.predict(training_data[test]))

    print(f'Acurácia: {accuracy_tacker.mean():.3f}, Desvio padrão: {accuracy_tacker.std():.3f}', end='\n\n')
    print(accuracy_tacker)
    return(accuracy_tacker.mean(), accuracy_tacker.std(), predictions, prediction_targets)


# %%
# Modelo Pequeno
print('Modelo Pequeno com de 1.7k parâmetros')
results = train(factory.small)

# %%
# Modelo Médio
print('Modelo Médio com de 3.5k parâmetros')
results = train(factory.medium)

# %%
# Modelo Grande
print('Modelo Grande com de 5.2k parâmetros')
results = train(factory.large)
