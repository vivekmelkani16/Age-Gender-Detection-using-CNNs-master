from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
import math

# initial parameters
epochs = 100
learning_rate  = 1e-3 #learning Rate
batch_size = 64
img_dims = (96,96,3)

data = []
labels = []

# load image files from the dataset
image_files = [f for f in glob.glob(r'C:\pr\MainGender-Detection\MainGender-Detection\gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

# converting images to arrays and labelling the categories
for img in image_files:

    image = cv2.imread(img)
    
    image = cv2.resize(image, (img_dims[0],img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2] # 'C:\pr\MainGender-Detection\MainGender-Detection\gender_dataset_face\woman\face_1162.jpg
    if label == "woman":
        label = 1
    else:
        label = 0 #man
        
    labels.append([label]) # [[1], [0], [0], ...]

#SVD

# pre-processing
data = np.array(data, dtype="float") / 255.0 #if image size is big then resize it
labels = np.array(labels)

# split dataset for training and validation(testing)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)
# 0.2 testing model 0.8 train model
#performs a random split (random_state=42)
#use same data for testing and training

trainY = to_categorical(trainY, num_classes=2) # [[1, 0], [0, 1], [0, 1], ...]
testY = to_categorical(testY, num_classes=2)
"""
to_categorical to transform your training data(Converts a class vector (integers) 
to binary class matrix
here data divide into 2 classes)
"""

# augmenting dataset(transfrom image in different-different shape) 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# define model
def build(width, height, depth, classes):
    model = Sequential()
   
    """Sequential() is the easiest way to build a model in Keras. 
    It allows you to build a model layer by layer."""

    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first": 
        #Returns a string, either 'channels_first' or 'channels_last'
        #it defines where the 'channels' dimension is in the input data.
        inputShape = (depth, height, width)
        chanDim = 1
    
    """
    The axis that should be normalized, after a Conv2D layer with
    data_format="channels_first", 
    set axis=1 in BatchNormalization.
    """

    # add() function to add layers to our model.
    #Our first 2 layers are Conv2D layers. These are convolution layers
    # that will deal with our input images, which are seen as 2-dimensional matrices.
    #32 or 64 the number of nodes in each layer
    #Here we are learning a total of 32/64/128 filters and then we use Max Pooling to reduce the spatial dimensions of the output volume.

    #Kernel size is the size of the filter matrix for our convolution. 
    #So a kernel size of 3 means we will have a 3x3 filter matrix. 
   
    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    
    #2 filters. Max pooling is then used to reduce the spatial dimensions of the output volume.
    #our output spatial volume is decreasing our number of filters learned is increasing

    model.add(Activation("relu")) #relu is a type of activation function f(x)=max(0,x)
    """The activation function we will be using for our first 2 layers is the ReLU.
        The rectified linear activation function or ReLU for short is a piecewise linear 
         function that will output the input directly if it is positive, otherwise, it will output zero.
  """

    model.add(BatchNormalization(axis=chanDim))
    """The keras BatchNormalization layer uses axis=-1(normalize your data by columns) 
    as a default value and states that the feature axis is typically normalized.
    Batch normalization applies a transformation 
    that maintains the mean output close to 0 and the output standard deviation close to 1. 
    """

    """The pooling operation involves sliding a two-dimensional filter over each channel 
    of feature map and summarising the features lying within the region covered by the filter. 
    
    Max pooling is a pooling operation that selects the maximum element from the region 
    of the feature map covered by the filter. 
    Thus, the output after max-pooling layer would be a
     feature map containing the most prominent features of the previous feature map.
     There are some other pooling layer Average Pooling 
     pool_size: tuple of 2 integers, factors by which to downscale (vertical, horizontal). 
     (3, 3) will halve the image in each dimension.
     """
    
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))
    #Dropout layers they prevent overfitting on the training data.



    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #Flatten serves as a connection between the convolution and dense layers.
    #Flatten() method converts multi-dimensional matrix to single dimensional matrix.

    """
    Dense is the layer type we will use in for our output layer.
    Dense Layer is simple layer of neurons in which each neuron receives 
    input from all the neurons of previous layer, thus called as dense. 
    Dense Layer is used to classify image based on output from convolutional layers. 
    1024 nuron node
    """

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))
    """
    The input to the function is transformed into a value between 0.0 and 1.0. 
    also called the logistic function
    Inputs that are much larger than 1.0 are transformed to the value 1.0, similarly, 
    values much smaller than 0.0 are snapped to 0.0.
    """

    return model

# build model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
                            classes=2)

# compile the model
# define a learning rate schedule instead of "decay"
lr_schedule = ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=len(trainX) // batch_size,  # decay per epoch
    decay_rate=0.96,                        # adjust as needed
    staircase=True
)

opt = Adam(learning_rate=lr_schedule)
#use for weight updation
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=math.ceil(len(trainX) / batch_size),  # use ceil to include all images
    epochs=epochs,
    verbose=1
)

# save the model to disk
model.save('gender_detection.keras')   # recommended for Keras 3

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
#validation_loss
plt.plot(np.arange(0,N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig('plot.png')