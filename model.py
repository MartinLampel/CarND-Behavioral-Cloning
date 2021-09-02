
import cv2
import csv
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from math import ceil
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from sklearn.model_selection import train_test_split


batch_size = 32
steering_correction = [.25, 0., -.25]

def load_samples(path):
    samples = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    
    return samples
        
        
def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
  
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

        
    return model

def generator(samples, batch_size=16):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
              
                selection = np.random.randint(len(steering_correction))
                name = batch_sample[selection]
                image = cv2.imread(name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                               
                
                angle = float(batch_sample[3]) + steering_correction[selection]
                
                images.append(image)
                angles.append(angle)
                
                 # Flipping
                images.append(cv2.flip(image,1))
                angles.append(angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield X_train, y_train

def plot_loss(history_object):
    
    plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('loss.jpg')
            
            
samples = load_samples('./driving_log.csv')
model = create_model()
model.summary()

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, \
            steps_per_epoch=ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=ceil(len(validation_samples)/batch_size), \
            epochs=3, verbose=1)


model.save('model.h5')

plot_loss(history_object)