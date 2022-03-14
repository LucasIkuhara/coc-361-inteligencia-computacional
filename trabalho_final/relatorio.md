# Relatório Final da disciplina de  Inteligência Computacional - COC 361

de Lucas Ribeiro Ikuhara - DRE 119019172.

## Introdução

### Descrição do problema

Dado o perfil de um consumidor, baseado em informações diversas a seu respeito, como escolaridade, estado marital e compras anteriores prever o resultado de uma campanha publicitária que busca vender um produto.

A proposta é a construção de um modelo utilizando redes neurais para tentar prever o resultado de uma campanha para um determinado indivíduo, a partir de seus dados.

Além disso, buscaremos avaliar os resultados obtidos e comparar-los com outros métodos mais tradicionais, como árvores de decisão e SVMs.

## Dataset e Tecnologia

### Descrição dos dados

O dataset consiste em dados de 2.240 clientes, com 28 colunas. As features presentes são de divididas entre qualitativas, quantitativas e binárias.
As informações dizem respeito as características dos receptores campanhas publicitárias em relação três principais aspectos:

#### Perfil

Diz respeito a sua vida, como por exemplo, ano de nascimento, nível de educação, estado civil, país, renda e a presença ou ausência de crianças ou adolescentes em casa.

#### Atividade no site

Diz respeito aos seus padrões de consumo e atividade na plataforma, como data de criação da conta, tempo desde a última compra, soma de valores de produtos comprados por categoria, número de compras com e sem descontos, e reclamações passadas.

#### Histórico de respostas a campanhas

Diz respeito a reação do cliente a campanhas publicitárias passadas. Denota se as últimas cinco campanhas geraram uma compra, individualmente.

#### Apresentação por coluna

| Coluna | Descrição | Tipo |
|---|---|---|
| ID | Id único do usuário | Nominal
| Year_Birth | Ano de nascimento do usuário | Quantitativa
| Country | Localização do usuário | Qualitativa
| Education | Nível de educação | Qualitativa
| Marital_Status | Estado Civil do usuário | Qualitativa
| Income | Renda anual da residência do usuário | Quantitativa
| Kidhome | Quantidade de crianças que moram com o usuário | Quantitativa
| Teenhome | Quantidade de adolescentes que moram com o usuário | Quantitativa
| Dt_Customer | Data de criação da conta | Quantitativa
| Recency | Número de dias desde a última compra | Quantitativa
| MntWines | Valor gasto em vinhos no últimos 2 anos | Quantitativa
| MntFruits | Valor gasto em frutas no últimos 2 anos | Quantitativa
| MntMeatProducts | Valor gasto em carne no últimos 2 anos | Quantitativa
| MntFishProducts | Valor gasto em peixes no últimos 2 anos | Quantitativa
| MntSweetProducts | Valor gasto em doces no últimos 2 anos | Quantitativa
| MntGoldProds | Valor gasto em ouro no últimos 2 anos | Quantitativa
| NumDealsPurchases | Número de compras feitas com descontos | Quantitativa
| NumWebPurchases | Número de compras realizadas através do site | Quantitativa
| NumCatalogPurchases | Número de compras feitas usando um catálogo | Quantitativa
| NumStorePurchases | Número de compras feitas em lojas físicas | Quantitativa
| NumWebVisitsMonth | Número de visitas ao site por mês | Quantitativa
| AcceptedCmp1 | Denota se a primeira campanha de marketing anterior recebida gerou ou não uma compra | Quantitativa (binária)
| AcceptedCmp2 | Denota se a segunda campanha de marketing anterior recebida gerou ou não uma compra |  Quantitativa (binária)
| AcceptedCmp3 | Denota se a terceira campanha de marketing anterior recebida gerou ou não uma compra | Quantitativa (binária)
| AcceptedCmp4 | Denota se a quarta campanha de marketing anterior recebida gerou ou não uma compra | Quantitativa (binária)
| AcceptedCmp5 | Denota se a quinta campanha de marketing anterior recebida gerou ou não uma compra | Quantitativa (binária)
| Response | Denota se a campanha de marketing recebida gerou ou não uma compra |  Quantitativa (binária)
| Complain | Denota se o consumidor fez alguma reclamação nos últimos 2 anos |  Quantitativa (binária)

### Tecnologias

Todo o código foi desenvolvido utilizando a linguagem Python 3, empregando diversas bibliotecas diferentes para funcionalidade específicas:

- Pandas: utilizado para a leitura e manipulação dos dados recebidos em csv.
- Plotly: utilizado para gerar todas as visualizações apresentadas.
- Scikit Learn (sklearn): utilizado para o treinamento de todos os modelos menos a rede MLP. Utilizado também para a avaliação de modelos e validação cruzada.
- Tensorflow & Keras: utilizado para gerar o modelo classificador MLP.

## Metodologia

### Apresentação da solução

Serão usados diversos modelos de classificação de caráter linear e não-linear para tentar prever a variável alvo 'Response' com os dados de entrada de um usuário, classificando entre venda e não-venda. A classe venda significa que o usuário aceita uma campanha publicitária e realiza uma compra, enquanto a classe não-venda indica que a compra não ocorreu.

### Descrição teórica

Os modelos de classificação escolhidos foram divididos em dois grupos, os lineares e os não-lineares. Os modelos explicados são suas versões utilizadas no trabalho. Desta maneira, em modelos como a random forest, que pode ser usada para classificação ou regressão, o enfoque da explicação será o modelo de classificação, e seu funcionamento, e não todas as suas possíveis variações e empregos.

#### Lineares

##### Regressão Logística

A regressão logística é um método que consiste utilizar um curva sigmoide para
representar a probabilidade de uma dada entrada se encontrar um uma determinada classe. Para isso, é necessário no passo de treino, encontrar os parâmetros que
descrevem a sigmoide que melhor separa classes alvo. O modelo final, tomará a forma de uma função $F(x)\rightarrow y$, onde o y define a probabilidade [0,1] da entrada ser parte de uma determinada classe. No caso de um classificador entre apenas duas classes, pode-se interpretar o y = 0 como sendo uma classe,
e y = 1 sendo a segunda. Dessa maneira, vamos supor que uma entrada x=4 tenha 10% de chance de ser da classe 0 e 90% de chance de ser da classe 1. O modelo de duas classes quando treinado, deverá se comportar da seguinte maneira>
$$F(4) = 0.9$$
Assim a chance de pertencer a classe 0 será dada por $1 - F(4) = 1 - 0.9 = 0.1$, enquanto sua chance de pertencer a classe 1, é simplesmente $F(4) = 0.9$.

A função $F(x)$, como mencionado, é descrita por uma curva sigmoide, da seguinte forma:

$$ F(x) = \frac{1}{1-e^{-\lambda x}}$$

Por isso, o processo de treinamento desse modelo consiste em encontrar o valor de $\lambda$ que resulta nos melhores valores de uma métrica predefinida, como acurácia, por exemplo.

##### Classificação Bayesiana

WIP

#### Não-Lineares

##### Árvore de Decisão

A Árvore de Decisão é um modelo de classificação baseado na combinação de diversas hipóteses discretas a respeito dos dados de entrada em uma estrutura de árvore binária. Dessa maneira, teremos que os nós internos da árvore são formados por condições simples que definem para qual nó filho os dados devem seguir para continuar a classificação. Esse processo, iniciado na raiz da árvore, se repete até que uma decisão resulte em um nó filho que é uma das folhas da árvore. Cada folha determinará uma classe, permitindo que n folhas representem a mesma classe simultaneamente. Ao atingirmos uma folha, a saída do modelo será a classe da folha em questão.

O processo de treinamento desse tipo de modelo consiste em alterar as condições dos nós internos da árvore, para que o modelo como um todo maximize uma métrica, como a acurácia por exemplo. Os nós possuem decisões tipicamente do tipo ```if (feature > param)```, logo o valor de param é modificado iterativamente, até que se chegue ao melhor resultado de classificação.

##### Random Forest

A Random Forest, ou Floresta Aleatória em português, é uma combinação de n árvores de decisão, que são treinadas com samples diferentes dos dados de entrada. Ao tentar classificar uma entrada, cada árvore de decisão chegará a um resultado diferente, e o resultado de maior incidência será considerado o resultado da random forest como um todo, em um processo conhecido como votação.

O treinamento desse modelo, consiste simplesmente no treinamento das diferentes árvores com partes diferentes dos dados de entrada, como descrito na seção de Árvores de Decisão.

O modelo de Random Forest é creditado com uma capacidade maior de generalização que uma única árvore de decisão, visto que são propensas a demonstrar problemas de overtfitting.

##### Rede Neural - Multi-layer perceptron

O modelo utilizado consiste na agregação de diversos perceptrons em camadas conetadas. Cada perceptron atua recebendo sinais de outros perceptrons e multiplicando cada sinal por um peso. Todos os sinais são somados e adicionados a um valor chamado de viés formando, gerando o valor de ativação daquele perceptron. Esse valor é usado como entrada para uma função de ativação, que diz se o perceptron deverá ou não disparar com a entrada dada. Ao combinarmos milhares desses perceptrons, conseguimos criar redes cada vez mais capacidade, que conseguem reter muito conhecimento nos seus pesos e viéses. Esse tipo de rede é conhecido por ter uma grande capacidade de generalização.

Para o treinamento de modelos desse tipo, é necessária uma função de perda, que calculará um score para os pesos atuais do modelo em relação a uma métrica alvo. A partir disso, um processo conhecido como back-propagation é empregado para alterar os valores dos pesos, que calcula através de derivadas a influência de cada perceptron nas ativações finais da rede e portanto nos resultados, até que os resultados sejam considerados aceitáveis.

## Resultados

### Procedimento de validação

Todos os resultados obtidos foram obtidos através do procedimento de validação cruzada de 10 folds. Isso significa que o dataset foi dividido em 10 partes iguais de maneira determinística, e foram usadas 9 por vez para treino e uma para teste. O processo é repetido mudando qual das partes (ou folds) estão sendo usadas para teste.

Ao final do ciclo de treinamento de cada fold, os dados de teste são usado para gerar métricas, como verdadeiros positivos, veradadeiros negativos, falsos negativos, falsos positivos e acurácia.

Após o processo ser feito 10 vezes, a acurácia média dos folds e seu desvio padrão são calculados.

#### Tratamento e visualização de dados

##### Dados faltantes, inválidos ou insignificantes

Os dados faltantes e inválidos, como nulos e NaNs foram removidos, trazendo o número de amostras de 2240 para 2216.

Além disso, a coluna ID foi inteiramente removida, pois não foi considerada relevante para a análise.

##### Visualização de dados

Começamos com a visualização inicial dos dados sem nenhuma forma de tratamento, visando principalmente descobrir quais colunas precisaram de tratamento. A forma de visualização escolhida foi o histograma de frequência, usando um gráfico auxiliar de violino para representar as distribuições.

Foram encontrados valores de outliers óbvios nos anos de nascimento e estado civil:

![Nascimento](./graphs/year.png)
![Estado Civil](./graphs/marital.png)

Esses valores foram descartados.

Foram também encontrados valores quer precisaram de formatação adicional, como por exemplo, datas e salários (formatadas como string com '$,.')

![Datas](./graphs/Datas.png)
![Income](./graphs/Income.png)

esses valores foram transformados em valores numéricos.

Alguns outros valores possuíam valores muito esparsos, como os valores gastos por categorias:

![amt](./graphs/log_ex.png)

Esses valores foram tratados com uma função log.

Por último, as variáveis qualitativas como estado civil e país sofreram de hot-encoding para que pudessem ser usadas para treinar os modelos.

##### Tratamento de dados

Após identificados todos os tratamentos necessários no passo anterior, os dados foram tratados usando pandas. Além dos tratamentos descritos, ao final todas as colunas foram standartizadas, e o novo dataframe, pronto para treinamento, foi salvo.

Após o tratamento, as features foram plotadas novamente. Exemplos pós tratamento:

![income ajustado](./graphs/income_ajustado.png)
![education ajustado](./graphs/education_ajustado.png)
![num ajustado](./graphs/num_ajustado.png)
![log ajustado](./graphs/log_ajustado.png)

##### Correlação

Para visualizar a correlação das variáveis, foi usada a seguinte matriz de correlação.

![Matriz de Correlaçãos](./graphs/matrix%20de%20correla%C3%A7%C3%A3o.png)

Quase nenhum par apresenta módulo da correlação maior que 0.5, com notáveis exceções como compras feitas com catálogo e compras de carne. No entanto, não considerou-se que nenhuma das variáveis era redundante, e portanto, nenhuma foi removida.

##### Resultados dos modelos lineares

###### Resultados: Regressão Logística

Para esse modelo, variou-se o solver utilizado entre LBFGS e SAGA, ambos com valores de 5000 iterações máximas, para que ambos consigam convergir.

- LBFGS

        Acurácia: 0.724, 
        Desvio padrão: 0.145
        Precisão: 0.700
        Recall: 0.541
        F1 Score: 0.610

Matriz de confusão

![lbfgs](./graphs/conf_matrix/reg_log_lbfgs.png)

- SAGA

        Acurácia: 0.724
        Desvio padrão: 0.145
        Precisão: 0.700
        Recall: 0.541
        F1 Score: 0.610

Matriz de confusão

![saga](./graphs/conf_matrix/reg_log_saga.png)

###### Resultados: Classificador Bayesiano

Os modelo utilizado foi um classificador Gaussiano (Gaussian Naive Bayes), e foi testado com os valores de smoothing de $1^{-9}$ e $1^{-11}$.

- Valores de smoothing de $1^{-9}$

        Acurácia: 0.671
        Desvio padrão: 0.230
        Precisão: 0.617
        Recall: 0.668
        F1 Score: 0.642

Matriz de confusão

![e11](./graphs/conf_matrix/bayes_11.png)

- Valores de smoothing de $1^{-11}$

        Acurácia: 0.659
        Desvio padrão: 0.250
        Precisão: 0.606
        Recall: 0.687
        F1 Score: 0.644

Matriz de confusão

![e9](./graphs/conf_matrix/bayes_9.png)

##### Resultados dos modelos não-lineares

###### Resultados: Árvore de Decisão

As árvores de decisão foram testadas, variando os parâmetros de critério entre 'Gini' e 'Entropy', ambas com a profundidade ilimitada.

- Gini

        Acurácia: 0.486
        Desvio padrão: 0.270
        Precisão: 0.49064449064449067
        Recall: 0.735202492211838
        F1 Score: 0.5885286783042395

Matriz de confusão

![arv_gini](./graphs/conf_matrix/dec_tree_gini.png)

- Entropy

        Acurácia: 0.506
        Desvio padrão: 0.241
        Precisão: 0.504
        Recall: 0.694
        F1 Score: 0.584

Matriz de confusão

![arv_entropy](./graphs/conf_matrix/dec_tree_entropy.png)

###### Resultados: Gradient Boosting

Os modelos de Gradient Boosting foram testados variando os hiper-parâmetros da função de perda entre 'deviance' e 'exponential', com 100 estimadores cada.

- Deviance

        Acurácia: 0.533
        Desvio padrão: 0.245
        Precisão: 0.5233050847457628
        Recall: 0.7017045454545454
        F1 Score: 0.5995145631067961

![gb_dev](./graphs/conf_matrix/grad_boost_dev.png)

- Exponential

        Acurácia: 0.515
        Desvio padrão: 0.250
        Precisão: 0.510
        Recall: 0.708
        F1 Score: 0.593

Matriz de confusão

![gb_exp](./graphs/conf_matrix/grad_boost_exp.png)

###### Resultados: Support Vector Machines

Os modelos de SVM foram testados variando os parâmetros de C. Foram testados os valores C=1 e C=2.s

- Classificador com C=1

        Acurácia: 0.688
        Desvio padrão: 0.146
        Precisão: 0.677
        Recall: 0.522
        F1 Score: 0.589

Matriz de confusão

![svm_c1](./graphs/conf_matrix/svm_c1.png)

- Classificador com C=2

        Acurácia: 0.720
        Desvio padrão: 0.139
        Precisão: 0.701
        Recall: 0.530
        F1 Score: 0.604

Matriz de confusão

![svm_c2](./graphs/conf_matrix/svm_c2.png)

###### Resultados: Random Forest

Os modelos de Random Forest foram testadas, variando os hiper-parâmetros de critério entre 'Gini' e 'Entropy', ambas com 100 estimadores.

- Gini

        Acurácia: 0.600
        Desvio padrão: 0.196
        Precisão: 0.577
        Recall: 0.621
        F1 Score: 0.598

Matriz de confusão

![rf_gini](./graphs/conf_matrix/rand_forst_gini.png)

- Entropy

        Acurácia: 0.591
        Desvio padrão: 0.179
        Precisão: 0.571
        Recall: 0.612
        F1 Score: 0.591

Matriz de confusão

![rf_ent](./graphs/conf_matrix/rand_forst_entropy.png)

###### Resultados: Rede Neural

Foram testadas três diferentes topologias de rede neural, mais especificamente uma rede de perceptrons multi-camada (MLP). A rede foi implementada utilizando Keras, uma API de alto nível para o framework Tensorflow.

As topologias variam o número de camadas internas, todas do tipo denso, completamente conectadas, entre uma e três camadas.

- Uma camada

                Acurácia: 0.730
                Desvio padrão: 0.127
                Precisão: 0.708
                Recall: 0.535
                F1 Score: 0.609

Matriz de confusão

![1lay](./graphs/conf_matrix/layer_one.png)

- Duas camadas

                Acurácia: 0.700
                Desvio padrão: 0.146
                Precisão: 0.674
                Recall: 0.551
                F1 Score: 0.607

Matriz de confusão

![2lay](./graphs/conf_matrix/layer_two.png)

- Três camadas

                Acurácia: 0.683
                Desvio padrão: 0.125
                Precisão: 0.662
                Recall: 0.547
                F1 Score: 0.599

Matriz de confusão

![3lay](./graphs/conf_matrix/layer_three.png)

##### Discussão e comparação dos resultados

Reunindo os resultados de todos os modelos, chegamos a seguinte tabela:

Modelo|Acurácia média|Desvio Padrão|Precisão|Recall|F Score|
|---|---|---|---|---|---|
Regressão Logística (LBFGS)| 0.724| 0.145| 0.700| 0.541| 0.610
Regressão Logística (SAGA)| 0.724| 0.145| 0.700| 0.541| 0.610
Gaussian Naive Bayes (1^-9)| 0.671| 0.230| 0.617| 0.668| 0.642
Gaussian Naive Bayes (1^-11)| 0.659| 0.250| 0.606| 0.687| 0.644
Árvore de Decisão (Gini)| 0.486| 0.270| 0.490| 0.735| 0.588
Árvore de Decisão (Entropy)| 0.506| 0.241| 0.504| 0.694| 0.584
Gradient Boosting (Deviance)| 0.533| 0.245| 0.523| 0.701| 0.599
Gradient Boosting (Exponential)| 0.515| 0.250| 0.510| 0.708| 0.593
SVM (C=1)| 0.688| 0.146| 0.677| 0.522| 0.589
SVM (C=2)| 0.720| 0.139| 0.701| 0.530| 0.604
Random Forest (Gini)| 0.600| 0.196| 0.577| 0.621| 0.598
Random Forest (Entropy)| 0.591| 0.179| 0.571| 0.612| 0.591
Rede Neural (1-layer)| 0.730| 0.127| 0.708| 0.535| 0.609
Rede Neural (2-layer)| 0.700| 0.146| 0.674| 0.551| 0.607
Rede Neural (3-layer)| 0.683| 0.125| 0.662| 0.547| 0.599

## Conclusões

Podemos perceber que os modelos lineares e não-lineares tiveram performance comparável, sendo os três melhores modelos a regressão logistíca, ao SVM e a rede neural.

Dada a natureza do problema, a previsão de o resultado de uma campanha de publicidade a partir de dados do usuário, os valores de acurácia esperados não podem ser muito altos, principalmente sem informações a respeito do produto que está sendo anunciado. Portanto, considerou-se a performance razoável.

Comparando apenas os melhores modelos de cada, temos que:

Modelo|Acurácia média|Desvio Padrão|Precisão|Recall|F Score|
|---|---|---|---|---|---|
Regressão Logística (LBFGS)| 0.724| 0.145| 0.700| 0.541| 0.610
SVM (C=2)| 0.720| 0.139| 0.701| 0.530| 0.604
Rede Neural (1-layer)| 0.730| 0.127| 0.708| 0.535| 0.609

Podemos perceber que os modelos de SVM e Rede Neural tiveram métricas muito próximas. Dado que o dataset foi completamente balanceado, daremos precedência as métricas de acurácia e desvio padrão. Por isso, o modelo recomendado será o modelo de rede neural de uma camada, pois foi o que obteve maior acurácia e menor desvio padrão.

Vale lembrar entretanto que o tempo de treinamento do modelo de rede neural é muito mais longo, e o processo no geral foi mais complexo. Dessa forma, caso a aplicação tenha necessidades especiais relacionadas a quantidade de memória gasta, tempo de treino ou mesmo complexidade, a recomendação passa a ser o modelo de SVM, pois conseguiu obter performance muito similar, e é bem menos custoso.