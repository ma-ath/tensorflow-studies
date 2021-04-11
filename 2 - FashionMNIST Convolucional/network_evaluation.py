import json
import pandas
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

#   Resolve um bug de "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#

#   Aqui vamos fazer alguma brincadeirinhas com nosso modelo treinado.
#   Primeiro de tudo, precisamos carregar o modelo e os pesos do disco.
model = keras.models.model_from_json(open('architecture.json').read())
model.load_weights('model_checkpoint.hdf5')

#   Em seguida, o compilamos novamente
model.compile(  optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

#   Vamos ver nosso modelo novamente
model.summary()

#   Em seguida, vamos avaliar nosso modelo. Para isso, precisamos carregar novamente os dados de treinamento.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

#   Primeiro vamos avaliar o modelo em todo o dataset de teste
evaluation = model.evaluate(test_images, test_labels)

print("loss: ",evaluation[0])
print("accuracy: ",evaluation[1])

#   Agora, vamos fazer algumas predicoes. Vamos prever o que eh a imagem numero 10 do teste

#prediction = model.predict(np.expand_dims(test_images[10], 0))

#   Notar que tivemos que expandir a dimensao do nosso tensor para essa previsao. Isso ta explicado no
#   tutorial original (https://www.tensorflow.org/tutorials/keras/classification).
#   A predicao de saida eh simplesmente um tensor de 10 posicoes.

#print(prediction.shape)

#   Nos interessa saber qual a posicao de maior valor desse tensor (ou seja, oq a rede neural diz que a saida eh)
#   Fazemos isso com o metodo

#print(np.argmax(prediction))

#   Para traduzir esse numero ao nome da classe predita, vamos simplesmente criar uma listinha com todos os nomer de classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#   E assim fazemos

#print("Classe predita:\t", class_names[np.argmax(prediction)])
#print("Classe real:\t", class_names[test_labels[10]])

#   Vamos fazer esse processo para as 10 primeiras imagens

for img in range(10):
    prediction = model.predict(np.expand_dims(test_images[img], 0))
    print("Predicao para imagem #"+str(img))
    print("Classe predita:\t", class_names[np.argmax(prediction)])
    print("Classe real:\t", class_names[test_labels[img]])

#   Podemos ver no terminal que nossa rede tem uma acuracia muito boa!
#   No tutorial original, eles fazem algumas outras ferramentas de vizualizacao muito legais, mas eu vou parar por aqui
#   Agora quero so mostrar como podemos obter os graficos de treinamento das redes. Para isso, precisamos carregar nosso
#   arquivo de historico ".csv" salvo. Faremos entao

with open("fit_history.csv",'r') as data: 
   fit_history = pandas.read_csv(data, index_col=0)

#   Nosso pandas dataframe
print(fit_history)

#   E um codigo que ira plotar o grafico de treinamento
fit_history.plot()
plt.show()

#   Vamos terminar por aqui esse mini-tutorial. Da proxima vez vamos criar um callback no keras, e trabalhar com redes convolucionais