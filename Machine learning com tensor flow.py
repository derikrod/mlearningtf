#!/usr/bin/env python
# coding: utf-8

# In[12]:


#O objetivo desse código é treinar a maquina para reconhecer imagens de roupas da biblioteca MNIST utilizando o TensorFlow
# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Bibliotecas Auxiliares
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print(np.__version__)


# In[10]:


#baixando imagens da biblioteca MNIST
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[11]:


#como o nome das classes não são armazenadas no banco de dados precisamos guardar as referências em uma variável
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[13]:


#buscando quantas imagens tem no modelo de treinamento nesse caso tem 60000 imagens 28x28 px
train_images.shape


# In[15]:


#inspecionando imagem antes de começar o treinamento da máquina
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# In[17]:


train_images = train_images / 255.0

test_images = test_images / 255.0
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[18]:


#construindo a primeira camada de aprendizado (layer)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# In[19]:


#compilando o modelo 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[20]:


#ensinando a máquina a associar o nome das roupas (label) com as imagens
model.fit(train_images, train_labels, epochs=10)


# In[21]:


#avaliando o nível de acerto do modelo
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


# In[23]:


#agora com a máquina treinada vamos treinar o quão boa ela é em adivinhar os elementos (predição)
predictions = model.predict(test_images)


# In[24]:


predictions[0]


# In[25]:


#checando se a máquina acerta qual é a label correspondente à imagem de posição 0 
#a maquina nos disses que é a label de posiçao 9 "Ankle boot" está correto
np.argmax(predictions[0])


# In[27]:


#vamos instanciar algumas coisas para avaliarmos graficamente os resultados
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[28]:


#vamos olhar graficamente primeiro a máquina analisando a roupa da posição 0 mostrou 42% de certeza que é uma ankle boot
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


# In[29]:


#vamos testar uma posição aleatória aqui ela já não teve muita certeza ficou entre sandália e tênis
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


# In[32]:


#vamos testar alguns elementos de roupa pra ver o nível de acertos
# Plota o primeiro X test images, e as labels preditas, e as labels verdadeiras.
# Colore as predições corretas de azul e as incorretas de vermelho.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

