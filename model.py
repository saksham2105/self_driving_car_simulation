import os
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
np.random.seed(0)


def load_data():
    data_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_valid, y_train, y_valid


def build_model():
    model = Sequential([
    Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE),
    Conv2D(24, (5,5), activation='relu', strides=(2, 2)),
    Conv2D(36, (5,5), activation='relu', strides=(2, 2)),
    Conv2D(48, (5,5), activation='relu', strides=(2, 2)),
    Conv2D(64, (3,3), activation='relu', strides=(1, 1)),
    Conv2D(64, (3,3), activation='relu', strides=(1, 1)),
    Dropout(0.5),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(50, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1),
    ])
    
    model.summary()

    return model
    

def train_model(model, X_train, X_valid, y_train, y_valid, validation_steps=None):
    #Saves the model after every epoch.
    #quantity to monitor, verbosity i.e logging mode (0 or 1), 
    #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
    
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    model.fit_generator(batch_generator('data', X_train, y_train, 40, True),
                        50,
                        10,
                        validation_data=batch_generator('data', X_valid, y_valid, 40, False),
                        validation_steps=20,
                        callbacks=[checkpoint],
                        verbose=1)
  

def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars().items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)
    
    #load data
    data = load_data()
    #build model
    model = build_model()
    #train model on data, it saves as model.h5 
    train_model(model, *data)


if __name__ == '__main__':
    main()