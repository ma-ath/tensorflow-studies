#   Codigo de treinamento para a nossa rede.

#   Importa coisa necessarias E nosso arquivo de modelos
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas
import os
from network_architectures import network_model

#   Primeira parte - fazer download e entender os dados ------------------------

#   Faz o download do fashion_mnist da base de dados do keras
fashion_mnist = keras.datasets.fashion_mnist

#   Carrega esse dataset, ja separando entre dados de treinamento e dados de teste
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#   Podemos verificar o formato desses arquivos
print("\nForma dos tensores de entrada:")
print("\ttrain_images.shape ",train_images.shape)
print("\ttest_images.shape ",test_images.shape)
print("\nForma dos tensores de saida:")
print("\ttrain_labels.shape ",train_labels.shape)
print("\ttest_labels.shape ",test_labels.shape)

#   Segunda parte - processar os dados -----------------------------------------
#   Em seguida, precisamos processar esses dados. Nesse caso, o processamento se resume a
#   limitar o valor de cada pixel das imagens no intervalo [0,1]. Fazemos isso dividindo por 255
train_images = train_images / 255.0
test_images = test_images / 255.0

#   Terceira parte - Definir o modelo e fazer o treinamento --------------------

#   Carregamos a arquitetura do nosso modelo do nosso arquivo de arquiteturas. Isso deixa o codigo
#   bem modular e limpo
model = network_model()

#   Compilamos o nosso modelo. Compilar significa dizer qual a funcao custo e otimizador vamos utilizar (entre outras coisinhas)

#   Nesse caso, o otimizador eh o "Adam". A funcao custo eh uma funcao de crossentropia, ja que estamos fazendo uma tarefa de classificacao (existem outras, podemos testar!)
#   Vamo monitorar a acuracia do nosso modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#   Podemos ver um resumo do nosso modelo com esse comando aqui
print("Resumo do modelo")
model.summary()

#   Agora vamos enfim treinar o modelo. Usamos o metodo "fit", passando como parametro nossos dados de treino, teste, e por quantas epocas
#   Esse metodo retorna os dados mostrados durante o treinamento, e em geral e interessante salva-los.
fit_history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=15)

#   Quarta parte - Salvar o modelo e outros dados --------------------
#   Neste ponto, nosso modelo ja esta treinado. Em geral, so queremos treinar o modelo uma unica vez, pois demora muito.
#   Vamos entao salvar esse modelo, junto com os dados de treinamento, no disco. Assim, se precisarmos usar ele novamente,
#   so precisamos carrega-lo.

#   Vamos primeiro salvar o historico de treinamento. Para isso, criaremos um dataframe no pandas (por ser mais facil de trabalhar assim)
fit_history_df = pandas.DataFrame(fit_history.history)

#   Com isso podemos salvar esses dados no disco diretamente
with open('fit_history.csv', mode='w') as f:
    fit_history_df.to_csv(f)
#   Reparar que um arquivo csv foi salvo no disco. Da uma olhadinha nesse arquivo depois

#   Em seguida, vamos salvar o modelo. Para isso, salvamos tanto a arquitetura quanto os pesos calculados.
#   A arquitetura do modelo podemos salvar como um arquivo json, com a funcao
model_json_string = model.to_json()
open('architecture.json', 'w').write(model_json_string)

#   Eh importante notar que estamos salvando a ULTIMA EPOCA do modelo, e que ela nao necessariamente sera a melhor.
#   Para salvarmos a melhor epoca do modelo, precisamos configurar um "callback" no keras. Faremos isso em outro tutorial

#   Agora um arquivo json foi salvo. Da uma olhadinha nesse arquivo tbm
#   Ta na hora de salvar os pesos. Isso eh feito com a funcao
model.save_weights('model_weights.h5', overwrite=True)

#   Agora esse arquivo .h5 tbm foi salvo no disco. Com isso temos todo o nosso modelo salvo no disco, e nao precisaremos treina-lo novamente
#   Nosso arquivo de treinamento acaba por aqui. Agora vamos analisar o modelo no outro arquivo