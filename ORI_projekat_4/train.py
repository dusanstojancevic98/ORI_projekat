from random import random, randint

import os

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


import cv2
import gc
from PIL import Image
import tensorflow as tf
from keras.utils import Sequence
from keras_preprocessing.image import ImageDataGenerator
from keras.initializers import Zeros, Ones, RandomNormal
from keras import Sequential
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, Callback, TerminateOnNaN
from keras.losses import mean_squared_error, mean_absolute_percentage_error, categorical_crossentropy
from keras.activations import linear, relu
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Dense, Dropout, ZeroPadding2D
import numpy as np
import xml.etree.ElementTree as ET

import pandas as p

epochs = 5
# steps = 50
# v_steps = 25
batch_size = 32

CLASSES = ["Normal", "Bacteria", "Virus"]

W = 480
H = 270


class PyDriveCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        with open("epoch-" + epoch + ".json", "w") as f:
            f.writelines(self.model.to_json())
        print("Uploaded model on epoch: {}".format(epoch))


class DataDivider:
    file_name: str
    _train: list
    _test: list
    _validate: list

    def __init__(self, file_name):
        self.file_name = file_name
        self._train = []
        self._test = []
        self._validate = []
        self._values = {}

    def divide_data(self):
        normal = []
        virus = []
        bacteria = []
        data = p.read_csv(self.file_name)
        for i, img in data.iterrows():
            # print("0, {}".format(img.iloc[0]))
            # print("1, {}".format(img.iloc[1]))
            # print("2, {}".format(img.iloc[2]))
            # print("3, {}".format(img.iloc[3]))
            # print("4, {}".format(img.iloc[4]))
            # if img.size > 5:
            #     print("5, {}".format(i, img.iloc[5]))

            name = img.iloc[1]
            if img.iloc[2] == "Normal":
                normal.append(name)
                self._values[name] = 0
            elif img.size > 3 and img.iloc[4] == "bacteria":
                bacteria.append(name)
                self._values[name] = 1
            elif img.size > 3 and img.iloc[4] == "Virus":
                virus.append(name)
                self._values[name] = 2
        print("Values: {}".format(self._values))
        self.__divide_array(normal)
        self.__divide_array(virus)
        self.__divide_array(bacteria)
        print("TRAIN: {}".format(len(self.train)))
        print("TEST: {}".format(len(self._test)))
        print("VALIDATE: {}".format(len(self._validate)))

    def __divide_array(self, array):
        n = len(array)
        print("n = {}".format(n))
        d = int(n * 7 / 10)
        d1 = int(n * 9 / 10)
        print("0-{}-{}-{}".format(d, d1, n))
        for i in range(0, d):
            self.train.append(array[i])
        for i in range(d, d1):
            self._test.append(array[i])
        for i in range(d1, n):
            self._validate.append(array[i])

    @property
    def train(self):
        return self._train

    @property
    def validate(self):
        return self._validate

    @property
    def test(self):
        return self._test

    @property
    def values(self):
        return self._values


class SegGenerator(Sequence):
    def __init__(self, images, outputs, batch, input_size=(160, 90), shuffle=True):
        self.images = images
        self.outputs = outputs
        self.shuffle = shuffle
        self.batch_size = batch
        self.indexes = np.arange(len(self.images))
        self.ep_cnt = 0
        self.b_cnt = 0
        self.input_size = input_size

    def __getitem__(self, index):
        if self.b_cnt == 0:
            print("START")
        s = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        self.b_cnt += 1
        print("{}/{} batches".format(self.b_cnt, len(self)))
        return self.getX(s), self.getY(s)

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.ep_cnt += 1
        self.b_cnt = 0
        print("Epoch {}".format(self.ep_cnt))

    def getX(self, s):
        x = np.empty((self.batch_size, self.input_size[1], self.input_size[0], 3))
        for index, i in enumerate(s):
            img = cv2.imread(self.images[i])
            img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255.
            if np.isnan(np.sum(img)):
                raise Exception("Nan in input! Image: '{}'", self.images[i])
            x[index] = img
        gc.collect()
        return x

    def getY(self, s):
        y = np.zeros((self.batch_size, len(CLASSES)))
        for index, i in enumerate(s):
            out = self.outputs[self.images[i]]
            y[index, out] = 1
        gc.collect()
        return y


accepted_diff = 0.01


#
# def linear_regression_equality(y_true, y_pred):
#     diff = K.abs(y_true - y_pred)
#     return K.mean(K.cast(diff < accepted_diff, tf.float32))


def train(model=None, input_size=(160, 90)):
    divider = DataDivider("chest_xray_data_set/metadata/chest_xray_metadata.csv")
    divider.divide_data()

    train_gen = SegGenerator(
        divider.train
        , divider.values
        , batch_size
        , input_size
    )
    val_gen = SegGenerator(
        divider.test
        , divider.values
        , batch_size
        , input_size
    )

    mcheckpoint = ModelCheckpoint('/content/drive/My Drive/model_chckpoint_{epoch:02d}.h5'
                                  , verbose=1
                                  , save_best_only=False
                                  , save_weights_only=False,
                                  mode='max'
                                  )
    if model is None:
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(90, 160, 3)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2), (2, 2)))
        model.add(Dropout(0.5))

        # 79 x 44
        model.add(Conv2D(64, (3, 3)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2), (2, 2)))
        model.add(Dropout(0.5))
        # 38 x 21

        # prebaceno

        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))

        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))

        # prebaceno
        model.add(Conv2D(512, (3, 3)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        # 36x19
        # 36 x 19
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Conv2D(1024, (3, 3), padding="same"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Conv2D(1024, (3, 5)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))

        # 32 x 17

        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Conv2D(512, (3, 3)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, (3, 3)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        # 28 x 13
        model.add(Conv2D(64, (3, 3)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        # 24 x 9
        """
        DENSE
        """
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.3))
        # model.add(Dropout(0.5))
        # model.add(Dense(256))
        # model.add(LeakyReLU(alpha=0.3))
        # model.add(Dropout(0.5))
        # model.add(Dense(128))
        # model.add(LeakyReLU(alpha=0.3))
        # model.add(Dropout(0.5))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0))
        model.add(Dropout(0.5))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Dense(16))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Dense(len(CLASSES), activation="softmax"))
        # 64 x 36
        model.compile(optimizer=Adam(0.01), loss=categorical_crossentropy, metrics=['accuracy'])
        model.summary()
    print("################ classes : {} ################".format(CLASSES))

    model.fit(train_gen, validation_data=val_gen, epochs=epochs
              , verbose=1
              , callbacks=[PyDriveCallback(), TerminateOnNaN()])
    model.summary()
    model.save('/content/drive/My Drive/model-small.h5')


if __name__ == '__main__':
    train()

