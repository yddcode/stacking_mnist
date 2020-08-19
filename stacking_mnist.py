
import os
import numpy as np
from PIL import Image
from keras.initializers import glorot_uniform
from keras.optimizers import SGD, Adam
from keras.layers import Input,Add,Dense,Conv2D,MaxPooling2D,UpSampling2D,Dropout,Flatten 
from keras.layers import BatchNormalization,AveragePooling2D,concatenate  
from keras.layers import ZeroPadding2D,add
from keras.layers import Dropout, Activation
from keras.models import Model,load_model
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, add, GlobalAvgPool2D
from keras.models import Model
from keras import regularizers
import keras
import numpy as np
from keras.models import Sequential
from keras import layers
from keras import backend as K
import sys

class Models(object):
    def __init__(self, data, path):
        self.data = data
        self.path = path
    def Data(self):
        if self.data == 'mnist':
            from keras.datasets import mnist
            (X_train, y_train), (X_test, y_test) = mnist.load_data() # 下载mnist数据集
            print(X_train.shape,y_train.shape) # 60000张28*28的单通道灰度图
            print(X_test.shape,y_test.shape)
            return X_train, y_train, X_test, y_test

        if self.data == 'printed':
            images = []
            # path = 
            for fn in os.listdir(self.path+'train'):
                if fn.endswith('.jpg'):
                    fd = os.path.join(self.path+'train',fn)
                    im = Image.open(fd).convert('L')
                    images.append(np.array(im))
            print('load success!')
            X_train = np.array(images)
            print (X_train.shape)
            y_train = np.loadtxt(self.path+'train label.txt')
            print (y_train.shape)

            images1 = []
            for fn in os.listdir(self.path+'test'):
                if fn.endswith('.jpg'):
                    fd = os.path.join(self.path+'test',fn)
                    im = Image.open(fd).convert('L')
                    images1.append(np.array(im))
            print('load success!')
            X_test = np.array(images1)
            print (X_test.shape)
            y_test = np.loadtxt(self.path+'test label.txt')
            print (y_test.shape)
            return X_train, y_train, X_test, y_test

        if self.data == 'handed':
            images = []
            for fn in os.listdir(self.path+'shouxietrain'):
                if fn.endswith('.jpg'):
                    fd = os.path.join(self.path+'shouxietrain',fn)
                    im = Image.open(fd).convert('L')
                    images.append(np.array(im))
            print('load success!')
            X_train = np.array(images)
            print (X_train.shape)
            y_train = np.loadtxt(self.path+'train.txt')
            print (y_train.shape)

            images1 = []
            for fn in os.listdir(self.path+'shouxietest'):
                if fn.endswith('.jpg'):
                    fd = os.path.join(self.path+'shouxietest',fn)
                    im = Image.open(fd).convert('L')
                    images1.append(np.array(im))
            print('load success!')
            X_test = np.array(images1)
            print (X_test.shape)
            y_test = np.loadtxt(self.path+'test.txt')
            print (y_test.shape)
            return X_train, y_train, X_test, y_test

    def Conv2d_BN(self, x, nb_filter, kernel_size, strides=(1,1), padding='same', name=None):  
        if name is not None:  
            bn_name = name + '_bn'  
            conv_name = name + '_conv'  
        else:  
            bn_name = None  
            conv_name = None  
    
        x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
        x = BatchNormalization(axis=3,name=bn_name)(x)  
        return x  
    
    def Conv_Block(self, inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):  
        x = self.Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')  
        x = self.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')  
        if with_conv_shortcut:  
            shortcut = self.Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)  
            x = add([x,shortcut])  
            return x  
        else:  
            x = add([x,inpt])  
            return x 

    def fit_model_ResNet34(self, trainX, trainy):
        input = Input(shape=(img_rows, img_cols, 1))  
        x = ZeroPadding2D((3,3))(input)  
        x = self.Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')  
        x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)  
        #(56,56,64)  
        x = self.Conv_Block(x,nb_filter=64,kernel_size=(3,3))  
        x = self.Conv_Block(x,nb_filter=64,kernel_size=(3,3))  
        x = self.Conv_Block(x,nb_filter=64,kernel_size=(3,3))  
        #(28,28,128)  
        x = self.Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
        x = self.Conv_Block(x,nb_filter=128,kernel_size=(3,3))  
        x = self.Conv_Block(x,nb_filter=128,kernel_size=(3,3))  
        x = self.Conv_Block(x,nb_filter=128,kernel_size=(3,3))  
        #(14,14,256)  
        x = self.Conv_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
        x = self.Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
        x = self.Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
        x = self.Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
        x = self.Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
        x = self.Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
        #(7,7,512)  
        x = self.Conv_Block(x,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
        x = self.Conv_Block(x,nb_filter=512,kernel_size=(3,3))  
        x = self.Conv_Block(x,nb_filter=512,kernel_size=(3,3))  
        # x = AveragePooling2D(pool_size=(2,2))(x)  
        x = AveragePooling2D(pool_size=(1,1))(x)  
        x = Flatten()(x) 
        x = Dense(num_classes,activation='softmax')(x)  
        
        model = Model(inputs=input,outputs=x)  
        sgd = SGD(decay=0.001,momentum=0.9)  
        model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])  
        # model.fit(trainX, trainy, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_to))
        return model

    def identity_block(self, X, f, filters, stage, block):
        """
        Implementation of the identity block as defined in Figure 4
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        
        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base   = "bn"  + str(stage) + block + "_branch"
        
        # Retrieve Filters
        F1, F2, F3 = filters
        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X
        
        # First component of main path
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid", 
                name=conv_name_base+"2a", kernel_initializer=glorot_uniform(seed=0))(X)
        #valid mean no padding / glorot_uniform equal to Xaiver initialization - Steve 
        
        X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
        X = Activation("relu")(X)
        ### START CODE HERE ###
        
        # Second component of main path (≈3 lines)
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
                name=conv_name_base+"2b", kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base+"2b")(X)
        X = Activation("relu")(X)
        # Third component of main path (≈2 lines)


        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                name=conv_name_base+"2c", kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base+"2c")(X)
        X = Add()([X, X_shortcut])
        X = Activation("relu")(X)
        ### END CODE HERE ###
        return X

    def convolutional_block(self, X, f, filters, stage, block, s = 2):
        """
        Implementation of the convolutional block as defined in Figure 4
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used
        
        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        # Save the input value
        X_shortcut = X

        ##### MAIN PATH #####
        # First component of main path 
        X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)
        
        ### START CODE HERE ###

        # Second component of main path (≈3 lines)
        X = Conv2D(F2, (f, f), strides = (1, 1), name = conv_name_base + '2b',padding='same', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(F3, (1, 1), strides = (1, 1), name = conv_name_base + '2c',padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        ##### SHORTCUT PATH #### (≈2 lines)
        X_shortcut = Conv2D(F3, (1, 1), strides = (s, s), name = conv_name_base + '1',padding='valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = layers.add([X, X_shortcut])
        X = Activation('relu')(X)
        
        ### END CODE HERE ###
        return X

    def ResNet50(self, input_shape, classes):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)
        
        # Stage 1
        X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv",
                kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name="bn_conv1")(X)
        X = Activation("relu")(X)
        X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
        X = self.identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")
        X = self.identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="c")
        ### START CODE HERE ###

        # Stage 3 (≈4 lines)
        # The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
        # The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
        X = self.convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=1)
        X = self.identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")
        X = self.identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
        X = self.identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")
        
        # Stage 4 (≈6 lines)
        # The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
        # The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
        X = self.convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
        X = self.identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")
        X = self.identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
        X = self.identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
        X = self.identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
        X = self.identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")
        
        # Stage 5 (≈3 lines)
        # The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
        # The 2 identity blocks use three set of filters of size [256, 256, 2048], "f" is 3 and the blocks are "b" and "c".
        X = self.convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
        X = self.identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")
        X = self.identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")
        
        # filters should be [256, 256, 2048], but it fail to be graded. Use [512, 512, 2048] to pass the grading
        
        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        # The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
        X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)
        
        ### END CODE HERE ###
        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation="softmax", name="fc"+str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
        # Create model
        model = Model(inputs=X_input, outputs=X, name="ResNet50")
        # model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def fit_model_ResNet50(self, trainX, trainy, input_shape, classes):
        model = self.ResNet50(input_shape=(img_rows, img_cols, 1), classes=num_classes)
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        # model.fit(trainX, trainy, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_to))
        return model

    def fit_model_ZFNet(self, trainX, trainy, input_shape, classes):
        model = Sequential()  
        model.add(Conv2D(96,(7,7),strides=(2,2),input_shape=input_shape,padding='valid',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
        model.add(Conv2D(256,(5,5),strides=(2,2),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
        model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        # model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
        model.add(MaxPooling2D(pool_size=(1,1),strides=(1,1)))  
        model.add(Flatten())  
        model.add(Dense(4096,activation='relu'))  
        model.add(Dropout(0.5))  
        model.add(Dense(4096,activation='relu'))  
        model.add(Dropout(0.5))  
        model.add(Dense(classes,activation='softmax'))  
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        # model.fit(trainX, trainy, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_to))
        return model

    def fit_model_LeNet(self, trainX, trainy):
        # define model
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation='tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Flatten())
        model.add(Dense(120, activation='tanh'))
        model.add(Dense(84, activation='tanh'))
        # model.add(Dense(10, activation='softmax'))　　#output
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        # fit model
        # model.fit(trainX, trainy, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_to))
        return model

    def fit_model_AlexNet(self, trainX, trainy):
        # define model
        model = Sequential()
        model.add(Conv2D(96, (11, 11), strides=(1, 1), input_shape=(img_rows, img_cols, 1), padding='same', activation='relu',
                    kernel_initializer='uniform'))
        # 池化层
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
        model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        #使用池化层，步长为2
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # 第三层卷积，大小为3x3的卷积核使用384个
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        # 第四层卷积,同第三层
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        # 第五层卷积使用的卷积核为256个，其他同上
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        # model.add(Dense(10, activation='softmax'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
        # fit model
        # model.fit(trainX, trainy, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_to))
        return model

    def fit_model_VGG16(self, x_train, trainy, x_test):
        # define model
        model = Sequential()
        model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(img_rows, img_cols, 1),padding='same',activation='relu',kernel_initializer='uniform'))
        model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        # model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Flatten())  
        model.add(Dense(4096,activation='relu'))  
        model.add(Dropout(0.5))  
        model.add(Dense(4096,activation='relu'))  
        model.add(Dropout(0.5))  
        # model.add(Dense(10, activation='softmax'))
        model.add(Dense(num_classes, activation='softmax'))
        # _x_train = []
        # _x_test = []
        #
        # for x_tr in x_train:
        #     _x_train.append(cv2.resize(x_tr, (224, 224)))
        # for x_te in x_test:
        #     _x_test.append(cv2.resize(x_te, (224, 224)))
        # print(np.array(_x_train).shape)
        # _x_train = np.array(_x_train).reshape(-1, 224, 224, 1)
        # _x_test = np.array(_x_test).reshape(-1, 224, 224, 1)
        # print('x_train shape:', _x_train.shape)
        # print(_x_train.shape[0], 'train samples')
        # print(x_test.shape[0], 'test samples')
        
        # sgd = SGD(decay=0.001,momentum=0.9)  
        # model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])  

        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        # model.fit(_x_train, trainy, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(_x_test, y_test_to))
        # model.fit(x_train, trainy, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test_to))
        return model

    def fit_model3(self, trainX, trainy):
        # define model
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        # fit model
        # model.fit(trainX, trainy, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_to))
        return model

    def fit_model4(self, trainX, trainy):
        # define model
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        # fit model
        # model.fit(trainX, trainy, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_to))
        return model

    def fit_model5(self, trainX, trainy):
        # define model
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        # fit model
        # model.fit(trainX, trainy, verbose=2, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_to))
        return model

    def fit_model6(self, trainX, trainy):
        # define model
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        # fit model
        # model.fit(trainX, trainy, verbose=2, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_to))
        return model

    def fit_model7(self, trainX, trainy):
        # define model
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
        # fit model
        # model.fit(trainX, trainy, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_to))
        return model

    def conv2d_bn(self, x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
        """
        conv2d -> batch normalization -> relu activation
        """
        x = Conv2D(nb_filter, kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            kernel_regularizer=regularizers.l2(0.0001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    def shortcut(self, input, residual):
        """
        shortcut连接，也就是identity mapping部分。
        """
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_height = int(round(input_shape[1] / residual_shape[1]))
        stride_width = int(round(input_shape[2] / residual_shape[2]))
        equal_channels = input_shape[3] == residual_shape[3]
    
        identity = input
        # 如果维度不同，则使用1x1卷积进行调整
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            identity = Conv2D(filters=residual_shape[3],
                            kernel_size=(1, 1),
                            strides=(stride_width, stride_height),
                            padding="valid",
                            kernel_regularizer=regularizers.l2(0.0001))(input)
        return add([identity, residual])
    
    def basic_block(self, nb_filter, strides=(1, 1)):
        """
        基本的ResNet building block，适用于ResNet-18和ResNet-34.
        """
        def f(input):
    
            conv1 = self.conv2d_bn(input, nb_filter, kernel_size=(3, 3), strides=strides)
            residual = self.conv2d_bn(conv1, nb_filter, kernel_size=(3, 3))
    
            return self.shortcut(input, residual)
        return f
    
    def residual_block(self, nb_filter, repetitions, is_first_layer=False):
        """
        构建每层的residual模块，对应论文参数统计表中的conv2_x -> conv5_x
        """
        def f(input):
            for i in range(repetitions):
                strides = (1, 1)
                if i == 0 and not is_first_layer:
                    strides = (2, 2)
                input = self.basic_block(nb_filter, strides)(input)
            return input
        return f
    
    def fit_model_resnet18(self, input_shape, classes):
        """
        build resnet-18 model using keras with TensorFlow backend.
        :param input_shape: input shape of network, default as (224,224,3)
        :param nclass: numbers of class(output shape of network), default as 1000
        :return: resnet-18 model
        """
        input_ = Input(shape=input_shape)
    
        conv1 = self.conv2d_bn(input_, 64, kernel_size=(7, 7), strides=(2, 2))
        pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)
    
        conv2 = self.residual_block(64, 2, is_first_layer=True)(pool1)
        conv3 = self.residual_block(128, 2, is_first_layer=True)(conv2)
        conv4 = self.residual_block(256, 2, is_first_layer=True)(conv3)
        conv5 = self.residual_block(512, 2, is_first_layer=True)(conv4)
    
        pool2 = GlobalAvgPool2D()(conv5)
        output_ = Dense(classes, activation='softmax')(pool2)
    
        model = Model(inputs=input_, outputs=output_)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        return model

from sklearn.model_selection import KFold
# from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    batch_size = 64
    num_classes = 10
    epochs = 60
    img_rows, img_cols = 28, 28

    models = Models(data='mnist', path='-')
    X_train, y_train, X_test, y_test = models.Data()

    if K.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(y_train.shape,y_test.shape)
    # convert class vectors to binary class matrices
    y_train_to = keras.utils.to_categorical(y_train, num_classes)
    y_test_to = keras.utils.to_categorical(y_test, num_classes)
    print(y_train_to.shape,y_test_to.shape)

    model_resnet18 = models.fit_model_resnet18(input_shape=(img_rows, img_cols, 1), classes=num_classes)
    model_resnet34 = models.fit_model_ResNet34(X_train, y_train_to)
    model_resnet50 = models.fit_model_ResNet50(X_train, y_train_to, input_shape=(img_rows, img_cols, 1), classes=num_classes)
    model_AlexNet = models.fit_model_AlexNet(X_train, y_train_to)
    model_LeNet = models.fit_model_LeNet(X_train, y_train_to)
    model_VGG16 = models.fit_model_VGG16(X_train, y_train_to, X_test)
    model_ZFNet = models.fit_model_ZFNet(X_train, y_train_to, input_shape=(img_rows, img_cols, 1), classes=num_classes)    
    model_3 = models.fit_model3(X_train, y_train_to)
    model_4 = models.fit_model4(X_train, y_train_to)
    model_5 = models.fit_model5(X_train, y_train_to)
    model_6 = models.fit_model6(X_train, y_train_to)
    model_7 = models.fit_model7(X_train, y_train_to)

    members = [model_resnet18, model_resnet34, model_resnet50, model_AlexNet, model_LeNet, model_VGG16, model_ZFNet, model_3, model_4, model_5, model_6, model_7]

    stack_train = np.zeros((X_train.shape[0], num_classes*len(members)),dtype=np.float32)  # Number of training data x Number of classifiers
    stack_test = np.zeros((X_test.shape[0], num_classes*len(members)),dtype=np.float32)  # Number of testing data x Number of classifiers
    n_folds = 5
    skf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    # For each classifier, we train the number of fold times (=len(skf))
    for j, clf in enumerate(members):
        print('Training classifier [%s]' % (j))
        for i, (train_index, cv_index) in enumerate(skf.split(X_train,y_train)):
            print('Fold [%s]' % (i))
        # for j,(train_index,test_index) in enumerate(skf.split(X_train,y_train)):
            tr_x = X_train[train_index]
            tr_y = y_train_to[train_index]
            print(tr_x.shape,tr_y.shape)
            # clf.fit(tr_x, tr_y)
            clf.fit(tr_x, tr_y, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_to))
            stack_train[cv_index, j*num_classes:(j+1)*num_classes] = clf.predict(X_train[cv_index])
            print('stack train', stack_train, stack_train.shape)
            stack_test[:, j*num_classes:(j+1)*num_classes] += clf.predict(X_test)
            print('stack test', stack_test, stack_test.shape)
    stack_test = stack_test / float(n_folds)
    print('stack test', stack_test, stack_test.shape)

    print('stack test', stack_test.shape)
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
    # 第二层模型
    clf_second = SVC()
    clf_second.fit(stack_train, y_train)
    pred = clf_second.predict(stack_test)
    print(pred.shape)
    print(accuracy_score(y_test,pred))

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')
    classify_report = classification_report(y_test, pred, digits=4)
    confusion_matrix = confusion_matrix(y_test, pred)

    output = sys.stdout
    outputfile = open("mnist_stacking_cnn.txt","a")
    sys.stdout = outputfile

    print('Stacked Test classify_report : \n', classify_report)
    print('Stacked Test confusion_matrix : \n', confusion_matrix)
    print('Stacked Test Accuracy: %.5f  \n' % acc)
    print('Stacked Test f1 score: %.5f  \n' % f1)
    print('Stacked Test recall score: %.5f  \n' % recall)

