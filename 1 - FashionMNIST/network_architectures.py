#   Neste arquivo, estaremos definindo qual o modelo de rede que usaremos para o treinamento
#   Eu acho interessante deixar o modelo em um arquivo separado, pois ajuda a modularizar o nosso codigo

from tensorflow import keras

#   Essa funcao simplesmente retorna o modelo que iremos treinar
def network_model():
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
    ])
    return model

#   Todo o codigo que esta dentro desse "if" so ira rodar caso vc execute esse arquivo (ou seja, ele seja o arquivo "__main__").
#   Ou seja, esse codigo nao ira rodar caso vc simplesmente importe esse arquivo em outro codigo python.
if __name__ == "__main__":
    #   Carrega o modelo
    model = network_model()

    #   Compila o modelo com otimizador "adam", funcao custo "sparse_categorical_crossentropy" e monitorando a metrica "accuracy".
    model.compile(  optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    
    #   Mostra um resumo do modelo, camada a camada e com numero de parametros.
    #   Isso e muito util para vermos se programamos o modelo corretamente.
    model.summary()