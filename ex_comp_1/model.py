# %%
# Importando pacotes
import pandas as pd
from sklearn import tree

# %%
# Ler dados
df = pd.read_csv('../star-dataset/Star3642_balanced.csv')
print('Sample data:')
print(df.tail())

# Separando Df de features e resultados 
training_targets = df['TargetClass']
training_data = df.drop('TargetClass', axis=1)
training_data = training_data.drop('SpType', axis=1)  # TODO Mapear os tipos para tipo int

# Treinar árvore de decisão
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(training_data, training_targets)

tree.plot_tree(decision_tree)
