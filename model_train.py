# code by Prerna Baranwal

import numpy as np
import cv2
import pandas as pd
import h5py

from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# constants
#load the validation, training and testing data as csv file
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_final_data.csv')
valid_data = pd.read_csv('valid_data.csv')


def change_brightness(image, bright_factor):
 # changes the brightness of the image   
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	# check for an alternative to the above
    # perform brightness augmentation only on the second channel
    hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
    
    # change back to RGB
    image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return image_rgb


def opticalFlowDense(image_current, image_next):
 # used to calculate the dense optical flow between a pair of images   
	# pairs of images used to calculated the dense optical flow
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)
    
    
    hsv = np.zeros((66, 220, 3))
    # set saturation
	# setting the saturation in the first channel 
	# check whether its the truth or not
    hsv[:,:,1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:,:,1]
 
    # Flow Parameters
    # flow_mat = cv2.CV_32FC2
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,  
                                        flow_mat, 
                                        image_scale, 
                                        nb_images, 
                                        win_size, 
                                        nb_iterations, 
                                        deg_expansion, 
                                        STD, 
                                        extra)
                                        
        
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  
        
    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    
    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    # convert HSV to float32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    
    return rgb_flow

# function to crop out the image
def preprocess_image(image):
 # crops the unnecessary portion of the image( the sky and the car stereo) and resizes to 66x220(required by the model being implemented)
 # input image is 640x480
 # this is performed on all the images being fed into the network   
    
    image_cropped = image[130:370, :] 
    
    image = cv2.resize(image_cropped, (220, 66), interpolation = cv2.INTER_AREA)
    
    return image

# function to convert the cropped image into a preprocessed one according to optical flow
def preprocess_image_valid_from_path(image_path, speed):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img)
    return img, speed

# function to preprocess the image according to the brightness factor
def preprocess_image_from_path(image_path, speed, bright_factor):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = change_brightness(img, bright_factor)    
    img = preprocess_image(img)
    return img, speed
# trainng data, validation data and testing data are all fed into pairs
#function to generate training data in sizes of batch 4
def generate_training_data(data, batch_size = 4):
    image_batch = np.zeros((batch_size, 66, 220, 3)) # nvidia input params
    label_batch = np.zeros((batch_size))
    while True:
        for i in range(batch_size):
            # generate a random index with a uniform random distribution from 1 to len - 1
            idx = np.random.randint(0, len(data))
            
            # Generate a random bright factor to apply to both images
            bright_factor = 0.2 + np.random.uniform()
            
               
            
            x1, y1 = preprocess_image_from_path(data.iloc[idx]['image_path_1'],
                                                data.iloc[idx]['speed_1'],
                                               bright_factor)
            
            # preprocess another image
            x2, y2 = preprocess_image_from_path(data.iloc[idx]['image_path_2'], 
                                                data.iloc[idx]['speed_2'],
                                               bright_factor)
           
            # compute optical flow send in images as RGB
            rgb_diff = opticalFlowDense(x1, x2)
                        
            # calculate mean speed
            y = np.mean([y1, y2])
            
            image_batch[i] = rgb_diff
            label_batch[i] = y
        
        yield shuffle(image_batch, label_batch)

def generate_validation_data(data):
    while True:
        for idx in range(0, len(data)): 
            
            x1, y1 = preprocess_image_valid_from_path(data.iloc[idx]['image_path_1'],data.iloc[idx]['speed_1'])
            x2, y2 = preprocess_image_valid_from_path(data.iloc[idx]['image_path_2'],data.iloc[idx]['speed_2'],)
            
            img_diff = opticalFlowDense(x1, x2)
            img_diff = img_diff.reshape(1, img_diff.shape[0], img_diff.shape[1], img_diff.shape[2])
            y = np.mean([y1, y2])
            
            speed = np.array([[y]])
            
            yield img_diff, speed


N_img_height = 66
N_img_width = 220
N_img_channels = 3

def nvidia_model():
    inputShape = (N_img_height, N_img_width, N_img_channels)

    model = Sequential()
    # normalization    
    # perform custom normalization before lambda layer in network
    model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = inputShape))

    model.add(Convolution2D(24, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv1'))
    
    
    model.add(ELU())    
    model.add(Convolution2D(36, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv2'))
    
    model.add(ELU())    
    model.add(Convolution2D(48, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv3'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, (3, 3), 
                            strides = (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv4'))
    
    model.add(ELU())              
    model.add(Convolution2D(64, (3, 3), 
                            strides= (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv5'))
              
              
    model.add(Flatten(name = 'flatten'))
    model.add(ELU())
    model.add(Dense(100, kernel_initializer = 'he_normal', name = 'fc1'))
    model.add(ELU())
    model.add(Dense(50, kernel_initializer = 'he_normal', name = 'fc2'))
    model.add(ELU())
    model.add(Dense(10, kernel_initializer = 'he_normal', name = 'fc3'))
    model.add(ELU())
    
    model.add(Dense(1, name = 'output', kernel_initializer = 'he_normal'))
    
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = adam, loss = 'mse')

    return model

filepath = 'model.h5'
earlyStopping = EarlyStopping(monitor='val_loss', 
                              patience=1, 
                              verbose=1, 
                              min_delta = 0.23,
                              mode='min',)
modelCheckpoint = ModelCheckpoint(filepath, 
                                  monitor = 'val_loss', 
                                  save_best_only = True, 
                                  mode = 'min', 
                                  verbose = 1,
                                 save_weights_only = True)
callbacks_list = [modelCheckpoint]

model = nvidia_model()

BATCH = 16

val_size = len(valid_data.index)
valid_generator = generate_validation_data(valid_data)
train_size = len(train_data.index)
train_generator = generate_training_data(train_data, BATCH)

history = model.fit_generator(
        train_generator, 
        steps_per_epoch = 400, 
        epochs = 100,
        callbacks = callbacks_list,
        verbose = 1,
        validation_data = valid_generator,
        validation_steps = val_size)

print(history)

model.save_weights("model.h5")
print("Saved model to disk")

#### testing the data to get the speed from the test video
test_size = len(test_data.index)
test_generator = generate_validation_data(test_data)

model.load_weights('/floyd/home/model.h5')
print("Loaded model from disk")
# evaluate loaded model on test data

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss = 'mse')
preds = model.predict_generator(test_generator,test_size, verbose=1)
np.savetxt('test.txt', preds, delimiter=',')

