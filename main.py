import numpy as np
import tensorflow as tf

# particular file for library fashion_mnist -> load_data()
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()


import matplotlib.pyplot as plt
# plt.imshow(x_train[0],cmap='hot')
# plt.axis('off')
# plt.show()


# Image is Flatten -> image are in single dimensions

x_train = x_train/255
x_test = x_test/255
x_train = x_train.reshape(60000,28,28,1)
x_test  = x_test.reshape(10000,28,28,1)

#
class_name = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
from tensorflow.keras.utils import to_categorical # One Hot-Encoding //

y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten

# Construction of basic simple CNN #
# Below Steps #

model = Sequential()
# Convolutional layer
model.add(Conv2D(filters = 32,kernel_size =(4,4),input_shape = (28,28,1),activation='relu'))
# Max Pooling
model.add(MaxPool2D(pool_size=(2,2)))
# Flatten
model.add(Flatten())
# Dense Layers
model.add(Dense(128,activation='relu'))
# Output Layer
model.add(Dense(10,activation='softmax'))
# Compile
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_cat_train,epochs=5)
i = 0
result = model.predict_classes(x_test[i].reshape(1,28,28,1))
print(result)
plt.imshow(x_test[i].reshape(28,28),cmap='gray')
plt.show()
print(class_name[result[0]])

