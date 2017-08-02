import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import os

import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

# Add nice styling to matplotlib and suppress scientific notation from numpy
np.set_printoptions(suppress=True)
sns.set_style('whitegrid')

# Network parameters. I like to have them all together for easy tuning
TEST_SIZE = 0.2
RANDOM_SEED = 453125
LR = 0.001
N_EPOCHS = 2
BATCH_SIZE = 100
IMG_WIDTH, IMG_HEIGHT = 82, 82

# Reading the data from train_labels.csv (not the best way to read all the images in the directory but it works).
df = pd.read_csv('train_labels.csv')

X = []
y = []

# function that converts images to numpy 3D Arrays
def img_to_array(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img

for name, label in tqdm(zip(df['name'], df['invasive'])):
    path = 'train/' + str(name) + '.jpg'
    X.append(img_to_array(path))
    if label==1:
        y.append([0, 1])
    else:
        y.append([1, 0])

# Converting X and y to numpy arrays of 3D numpy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Building the sequential model in Keras
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(82,82,3))) # A quick way to normalize the input images
model.add(Conv2D(16, (5,5), activation='relu', padding='same')) # 5x5 conv
model.add(MaxPooling2D(pool_size=(2,2)))	# max pooling, Image is now (41, 41)
model.add(Conv2D(32, (5,5), activation='relu', padding='same')) # 5x5 conv
model.add(MaxPooling2D(pool_size=(2,2)))	# max pooling, Image is now (20, 20)
model.add(Conv2D(64, (5,5), activation='relu', padding='same'))# 5x5 conv
model.add(Flatten()) # flattening the 3D filters (width, height, color_channels) down to 2D matrices to pass through the Dense layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5)) # Adding dropout is a regularization method that helps avoid overfitting.
model.add(Dense(2, activation='softmax')) # I use the softmax function for multi class classification with a probability distribution

# Setting the parameters for the Adam (Adaptive Moment Estimation) optimizer.
# The paper is worth taking a look at: https://arxiv.org/abs/1412.6980v8
opt = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# I also use data augmentation (mostly rotation and flipping) to ensure that the model generalizes to the test
# data since it is big enough to overfit the train set.
datagen = ImageDataGenerator(
            rotation_range = 35,
            width_shift_range = 0.22,
            height_shift_range = 0.23,
            shear_range = 0.21,
            zoom_range = 0.21,
            horizontal_flip = True,
            vertical_flip = True,
            fill_mode = 'nearest')

# Spliting the dataset to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# Fitting the data generator to the train images
datagen.fit(X_train)

# I load the weights from the best score I got so that I don't start from scratch
# feel free to comment out this part, download the dataset from kaggle and train the model from scratch
if os.path.isfile('checkpoints/best_weights.hdf5'):
	model.load_weights('checkpoints/best_weights.hdf5')
	print('Loaded Existing Model')

# Printing the model graph is always nice ;)
print(model.summary())
# Fitting the model to the train set
model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), callbacks=[ModelCheckpoint('model.{epoch:03d}-{val_loss:.3f}.hdf5')],validation_data=(X_test, y_test), steps_per_epoch = len(X)/100, epochs=N_EPOCHS, verbose=1)

# Saving the trained model
with open('model.json', 'w') as json_file:
        json_file.write(model.to_json())
model.save_weights('model.h5')
print('Model Saved!')

# It's time to make some predictions!

# Load the test images from sampel_submission (although not the best approach. It is a nice way to ensure that we read all the images)
df = pd.read_csv('sample_submission.csv')#
X = []
print('Loading test data')
for name in tqdm(df['name']):
    path = 'test/' + str(name) + '.jpg'
    X.append(img_to_array(path))

X = np.array(X, dtype=np.float32)

# Making a prediction on the entire test image dataset
pred = model.predict(X)

# Writing predictions to the csv format that kaggle wants
print('Making predictions')
with open('Submission.txt', 'w') as f:
    f.write('name,invasive\n')
    counter = 1
    for i in pred:
        line = str(counter) + ',' + str(i[1])
        f.write(line)
        f.write('\n')
        counter+=1