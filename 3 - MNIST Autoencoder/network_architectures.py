#   Neste arquivo, estaremos definindo qual o modelo de rede que usaremos para o treinamento
#   Eu acho interessante deixar o modelo em um arquivo separado, pois ajuda a modularizar o nosso codigo

import tensorflow as tf
from tensorflow import keras

#   Essa funcao simplesmente retorna o modelo que iremos treinar
def dense_autoencoder():
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(4, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(784, activation='sigmoid'))
    model.add(keras.layers.Reshape([28, 28]))
    return model

#   Todo o codigo que esta dentro desse "if" so ira rodar caso vc execute esse arquivo (ou seja, ele seja o arquivo "__main__").
#   Ou seja, esse codigo nao ira rodar caso vc simplesmente importe esse arquivo em outro codigo python.
if __name__ == "__main__":
    #   Carrega o modelo
    model = dense_autoencoder()

    #   Compila o modelo com otimizador "adam", funcao custo "sparse_categorical_crossentropy" e monitorando a metrica "accuracy".
    model.compile(  optimizer='adam',
                    loss='mse')
    
    #   Mostra um resumo do modelo, camada a camada e com numero de parametros.
    #   Isso e muito util para vermos se programamos o modelo corretamente.
    model.summary()
