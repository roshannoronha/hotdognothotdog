from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
#from keras import TensorBoard

#training/test data paths
train_data = "data/train"
test_data = "data/test"

##image preprocessing
#image size
img_width, img_height = 300,300
#rescale pixel values from [0-255] to [0-1]
datagen = ImageDataGenerator(rescale=1./255)

#create tensorboard object
#name = "hotdognothotdogCNN"
#tensorboard = TensorBoard(logdir= 'logs/{}'.format(name))

##image retrieval
#augment training data with randomly augmented images
train_datagen_augmented = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range= 0.2, horizontal_flip=True)

train_gen = train_datagen_augmented.flow_from_directory(train_data, target_size= (img_width, img_height), batch_size= 32, class_mode= 'binary')
#note that test gen batch size can be bigger since we are using it to test our model. Not used for training purposes.
test_gen = datagen.flow_from_directory(test_data, target_size= (img_width, img_height), batch_size= 32, class_mode= 'binary')

##define architecture
#the sequential model allows us to create a linear stack of layers
model = Sequential()

#there will be 3 convolution layers
model.add(Convolution2D(32,3,3, input_shape=(img_width,img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Convolution2D(32,3,3, input_shape=(img_width,img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Convolution2D(64,3,3, input_shape=(img_width,img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

#prevents overfitting
#increases the ability of a model to generalize
model.add(Dropout(0.5))

#the final layer only needs to have one node since we are only classifying hotdog/nothotdog
model.add(Dense(1))
model.add(Activation('sigmoid'))

#binary_crossentropy is the loss function that measures the inaccuracy of the prediction. binary_crossentropy used since we only have two categories. (backpropagation)
#the optimizer tweaks the weights of all the layers to minimize the error (gradient descent)
model.compile(loss = 'binary_crossentropy', optimizer= 'rmsprop', metrics = ['accuracy'])

##parameters for training the model
epoch = 40
train_samples = 31
test_samples = 15

model.fit_generator(train_gen, steps_per_epoch = train_samples, epochs = epoch, validation_data= test_gen, validation_steps = test_samples)

##save the weights
model.save('hotdogCNN2.h5')
print("NN2 saved to disk!")



