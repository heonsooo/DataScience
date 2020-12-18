import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical

#keras로 이미지 불러오기 
from keras.preprocessing.image import img_to_array, load_img, array_to_img

def plot_loss_curve(history):
    plt.figure(figsize = (5,3))
    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train','test'], loc = 'upper right')
    plt.show

def image_model():
    i,j,k = 1500,1500,1500
    #음식 데이터 to numpy  , Label = 0 Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    data_fd = []
    data_fd_label = []
    for num in range(1,i):
        #classes = ['exterior', 'interior','food']
        image_file = './images/{}{}.jpg'.format('food', num)
        image_array = img_to_array(load_img(image_file))
        #print(image_array)

        data_fd.append(image_array)
        data_fd_label.append(0)

    data_fd = np.array(data_fd)
    data_fd_label = np.array(data_fd_label)


    #실내 데이터 to numpy  , Label = 1
    data_in = []
    data_in_label = []
    for num in range(1,j):
        #classes = ['exterior', 'interior','food']
        image_file = './images/{}{}.jpg'.format('interior', num)
        image_array = img_to_array(load_img(image_file))
        # print(image_array)

        data_in.append(image_array)
        data_in_label.append(1)

    data_in = np.array(data_in)
    data_in_label = np.array(data_in_label)


    #실외 데이터 to numpy  , Label = 2
    data_ex= []
    data_ex_label= []
    for num in range(1,k):
        #classes = ['exterior', 'interior','food']
        image_file = './images/{}{}.jpg'.format('exterior', num)
        image_array = img_to_array(load_img(image_file))
        #print(image_array)
        data_ex.append(image_array)

        data_ex_label.append(2)

    data_ex = np.array(data_ex)
    data_ex_label = np.array(data_ex_label)


    train_test_split = 0.85

    train_data_fd = int(len(data_fd)*train_test_split)
    train_data_in = int(len(data_in)*train_test_split)
    train_data_ex = int(len(data_ex)*train_test_split)
    #print(train_data_fd,train_data_in,train_data_ex)
    #print(len(data_fd[:train_data_fd]))


    #  X_train , X_test 만들기 
    X_train_data_set = np.concatenate((data_fd[:train_data_fd], data_in[:train_data_in], data_ex[:train_data_ex]))
    # print(X_train_data_set.shape)
    X_test_data_set =np.concatenate((data_fd[train_data_fd: ], data_in[train_data_in:], data_ex[train_data_ex:]))
    # print(X_test_data_set.shape)

    y_train_data_set = np.concatenate((data_fd_label[:train_data_fd], data_in_label[:train_data_in], data_ex_label[:train_data_ex]))
    y_test_data_set = np.concatenate((data_fd_label[train_data_fd: ], data_in_label[train_data_in:], data_ex_label[train_data_ex:]))
    # print(y_train_data_set.shape)
    # print(y_test_data_set.shape)

    # print(X_train_data_set)


    from sklearn.utils import shuffle
    X_train_0, y_train_0 = shuffle(X_train_data_set, y_train_data_set, random_state=42)
    X_test_0, y_test_0= shuffle(X_test_data_set, y_test_data_set, random_state=42)
    
    return X_train_0 , y_train_0, X_test_0, y_test_0

print('12end')


def train_image_model():   
        
    X_train, y_train,X_test, y_test = image_model()

    X_train = X_train.reshape(len(X_train),300,300,3) 
    X_test = X_test.reshape(len(X_test),300,300,3)
    #     print(type(X_train), type(X_test))
    #     print(y_train.shape, y_test.shape)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    #     print(y_train.shape, y_test.shape)
    #print(y_train[0])
    model = Sequential([
                #고정
                Input(shape=(300,300,3), name='input_layer'),
                
                
                Conv2D(16, kernel_size=3, activation='selu', name='conv_layer1'),
                
                
                #Dropout(0.5)
                MaxPooling2D(pool_size=2),
                Conv2D(8, kernel_size=3, activation='selu', name='conv_layer2'),
                MaxPooling2D(pool_size=2),
                Flatten(),
                # Dense(20, activation='selu', name='FC_layer')
        
        
                #고정
                Dense(3, activation='softmax', name='output_layer')
            ])
    print('hheellood')
    model.summary()    
    
    # categorical_crossentropy -> sparse_categorical_crossentropy
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=3)
    plot_loss_curve(history.history)
    print(history.history)
    print("train loss=", history.history['loss'][-1])
    print("validation loss=", history.history['val_loss'][-1])    
    
    model.save('model-201814132')
    
    return model

if __name__ == '__main__' :
    train_image_model()

