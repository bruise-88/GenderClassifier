from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras import backend as K

#To initialise the Neural Network
class CNN:
    @staticmethod
    def building_CNN(width, height, depth, classes):
        classifier = Sequential()
        chanDim = -1

        if K.image_data_format() == "channels_first":
            chanDim = 1
        #Convolutional layer
        classifier.add(Convolution2D(32,(3,3),padding = 'same',input_shape = (64,64,3)))
        classifier.add(Activation("relu"))
        classifier.add(BatchNormalization(axis=chanDim))
        classifier.add(MaxPooling2D(pool_size=(2,2)))
        classifier.add(Dropout(0.25))
        
        classifier.add(Convolution2D(64,(3,3),padding = 'same',input_shape = (64,64,3)))
        classifier.add(Activation("relu"))
        classifier.add(BatchNormalization(axis=chanDim))
        classifier.add(MaxPooling2D(pool_size=(2,2)))
        classifier.add(Dropout(0.25))
        
        classifier.add(Flatten())
        classifier.add(Dense(128, activation = 'relu'))
       
        classifier.add(Activation("relu"))
        classifier.add(BatchNormalization())
        classifier.add(Dropout(0.5))
        
        classifier.add(Dense(classes))
        classifier.add(Activation("sigmoid"))
        return classifier