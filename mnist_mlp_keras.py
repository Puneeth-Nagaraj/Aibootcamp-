#import libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#load the data
(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape)
print(y_test.shape)

#print(x_train[0])
#plt.imshow(x_train[0],cmap='gray')
#plt.show()
#print(f"label is: {y_train[0]}")

#normalize 
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

#to categorical

print(f"before label is: {y_train[100]}")
y_train=to_categorical(y_train)
print(f"after label is: {y_train[100]}")
print("hi")
y_test=to_categorical(y_test)

#architecture
model=Sequential()
model.add(Flatten(input_shape=(28,28))) #1st layer
model.add(Dense(128,'relu')) #128 neurons, 2nd layer
model.add(Dense(10,'softmax'))#10 neurons , last layer

#compile 
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

#train
#result=model.fit(x_train,y_train,epochs=1,batch_size=64,validation_data=(x_test,y_test))#64 images will be sent to arch every time 
result=model.fit(x_train,y_train,epochs=10,batch_size=64,validation_split=0.2)

#evaluate
loss,accuracy=model.evaluate(x_test,y_test)
print(F"test loss:{loss}")
print(F"test accuracy:{accuracy}")
print(result.history.keys())
print(result.history.values())
print(result.history)

#visualization
plt.plot(result.history['val_accuracy'],label="validation accuracy",color='blue')
plt.plot(result.history['accuracy'],label="train accuracy",color='red')
plt.title("train_acccuracy vs val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plt.plot(result.history['val_loss'],label="validation loss",color='blue')
plt.plot(result.history['loss'],label="train loss",color='red')
plt.title("train_loss vs val_loss")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
print('###########')