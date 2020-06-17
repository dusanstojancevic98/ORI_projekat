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

CLASSES = ["Normal", "bacteria", "Virus"]

W = 480
H = 270


class PyDriveCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        with open("epoch-" + epoch + ".json", "w") as f:
            f.writelines(self.model.to_json())
        print("Uploaded model on epoch: {}".format(epoch))


class DataDivider:
    file_name: str
    train: list
    test: list
    validate: list

    def __init__(self, file_name):
        self.file_name = file_name
        self.train = []
        self.test = []
        self.validate = []

    def divide_data(self):
        normal = []
        virus = []
        bacteria = []
        data = p.read_csv(self.file_name)
        for i, img in data.iterrows():
            if img.iloc[2] == "Normal":
                normal.append(img.iloc[1])
            elif img.size > 3 and img.iloc[4] == "bacteria":
                bacteria.append(img.iloc[1])
            elif img.size > 3 and img.iloc[4] == "Virus":
                virus.append(img.iloc[1])

    def __divide_array(self, array):
        n = len(array)
        d = int(n*7/10)
        d1 = int(n*9/10)
        self.train.append([array[i] for i in range(0, d)])
        self.test.append([array[i] for i in range(d, d1)])
        self.validate.append([array[i] for i in range(d1, n)])



class SegGenerator(Sequence):
    def __init__(self, images, outputs, batch, input_size=(160, 90), output_size=(64, 36), shuffle=True, limit=None):
        self.classes_colors = []
        self.classes = []
        self.images = images
        self.outputs = outputs
        self.shuffle = shuffle
        self.batch_size = batch
        self.input_size = input_size
        self.output_size = output_size
        self.names = []
        self.names_dict = {}
        i = 0
        for name in os.listdir(images):
            self.names.append(name)
            self.names_dict[name] = i
            # print("Added pic {}".format(name))
            i += 1
            if limit is not None:
                if i >= limit:
                    break
        print("Added {} X pics".format(len(self.names)))
        self.outputs_names = [0] * len(self.names)
        for name in os.listdir(outputs):
            rname = name.split("_train_color")[0] + ".jpg"
            if rname in self.names_dict:
                index = self.names_dict[rname]
                # print("Added val pic {}".format(name))
                self.outputs_names[index] = name
        print("Added {} Y pics".format(len(self.outputs_names)))
        if len(self.names) != len(self.outputs_names):
            raise Exception("Input and output are not equal in size!")
        self.indexes = np.arange(len(self.names))
        self.ep_cnt = 0
        self.b_cnt = 0

    def load_classes(self, path):
        with open(path) as f:
            lines = f.readlines()
            print(len(lines))
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                elems = line.split(",")
                c = elems[0]
                # print(elems[4], elems[5], elems[6])
                r = int(elems[4].split("\"")[1])
                g = int(elems[5])
                b = int(elems[6].split("\"")[0])
                color = (r, g, b)
                # print(color)
                self.classes.append(c)
                self.classes_colors.append(color)
                # print("Dodata klasa {}".format(c))

    def __getitem__(self, index):
        if self.b_cnt == 0:
            print("START")
        s = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        self.b_cnt += 1
        print("{}/{} batches".format(self.b_cnt, len(self)))
        return self.getX(s), self.getY(s)

    def __len__(self):
        return int(np.floor(len(self.names) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.ep_cnt += 1
        self.b_cnt = 0
        print("Epoch {}".format(self.ep_cnt))

    def getX(self, s):
        x = np.empty((self.batch_size, self.input_size[1], self.input_size[0], 3))
        for index, i in enumerate(s):
            img = cv2.imread(os.path.join(self.images, self.names[i]))
            img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.
            if np.isnan(np.sum(img)):
                raise Exception("Nan in input!")
            x[index] = img
        gc.collect()
        return x

    def getY(self, s):
        y = np.zeros((self.batch_size, self.output_size[1], self.output_size[0], len(self.classes)))
        for index, i in enumerate(s):
            img = cv2.imread(os.path.join(self.outputs, self.outputs_names[i]))
            # cv2_imshow( img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2_imshow(img)
            # cv2.waitKey(20)
            # sleep(1000)
            shape = img.shape
            n = np.zeros((self.output_size[1], self.output_size[0], 3))
            dw = shape[1] // self.output_size[0]
            dh = shape[0] // self.output_size[1]
            for ij, j in enumerate(range(0, shape[0], dh)):
                if ij >= self.output_size[1]:
                    break
                for ik, k in enumerate(range(0, shape[1], dw)):
                    if ik >= self.output_size[0]:
                        continue
                    index1 = j + dh // 2
                    index2 = k + dw // 2
                    if index1 > shape[0]:
                        index1 = shape[0] - 1

                    if index2 > shape[1]:
                        index2 = shape[1] - 1
                    n[ij][ik] = img[index1, index2]
            count = 0
            for j in range(self.output_size[1]):
                for k in range(self.output_size[0]):
                    for ci, color in enumerate(self.classes_colors):
                        cell = n[j, k]
                        if int(cell[0]) == color[0] and int(cell[1]) == color[1] and int(cell[2]) == color[2]:
                            y[index, j, k, ci] = 1
                            count += 1
                            break
                            # print("Promenjeno {} celija od {}".format(count, self.output_size[0]*self.output_size[1]))
        gc.collect()
        return y

    def get_class_size(self):
        return len(self.classes)


accepted_diff = 0.01


#
# def linear_regression_equality(y_true, y_pred):
#     diff = K.abs(y_true - y_pred)
#     return K.mean(K.cast(diff < accepted_diff, tf.float32))


def trainsmall(model=None):
    train_gen = SegGenerator(
        "/content/drive/My Drive/seg/images/train", "/content/drive/My Drive/seg/color_labels/train"
        , batch_size
        , input_size=(160, 90)
        , limit=500
        , output_size=(24, 9))
    train_gen.load_classes("/content/drive/My Drive/seg/categories.csv")
    val_gen = SegGenerator(
        "/content/drive/My Drive/seg/images/val"
        , "/content/drive/My Drive/seg/color_labels/val"
        , batch_size
        , limit=100
        , input_size=(160, 90)
        , output_size=(24, 9))
    val_gen.load_classes("/content/drive/My Drive/seg/categories.csv")
    mcheckpoint = ModelCheckpoint('/content/drive/My Drive/model_chckpoint_{epoch:02d}.h5'
                                  , verbose=1
                                  , save_best_only=False
                                  , save_weights_only=False,
                                  mode='max'
                                  )
    classes = train_gen.get_class_size()
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
        model.add(Dense(classes, activation="softmax"))
        # 64 x 36
        model.compile(optimizer=Adam(0.01), loss=categorical_crossentropy, metrics=['accuracy'])
        model.summary()
    print("################ classes : {} ################".format(classes))

    model.fit(train_gen, validation_data=val_gen, epochs=epochs
              , verbose=1
              , callbacks=[PyDriveCallback(), TerminateOnNaN()])
    model.summary()
    model.save('/content/drive/My Drive/model-small.h5')


def train():
    model = Sequential()

    train_gen = SegGenerator("/content/drive/My Drive/seg/images/train",
                             "/content/drive/My Drive/seg/color_labels/train", batch_size)
    train_gen.load_classes("/content/drive/My Drive/seg/categories.csv")
    val_gen = SegGenerator("/content/drive/My Drive/seg/images/val", "/content/drive/My Drive/seg/color_labels/val",
                           batch_size)
    val_gen.load_classes("/content/drive/My Drive/seg/categories.csv")
    classes = train_gen.get_class_size()
    print("######################### classes : {} ################".format(classes))
    mcheckpoint = ModelCheckpoint('/content/drive/My Drive/model_chckpoint_{epoch:02d}.h5'
                                  , verbose=1
                                  , save_best_only=False
                                  , save_weights_only=False,
                                  mode='max'
                                  )
    # model.add(Conv2D(64, (3, 3), input_shape=(180, 320, 3)))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling2D((2, 2), (2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(64, (2, 2), input_shape=(90, 160, 3)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))

    # 159 x 89
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2), (2, 2)))
    model.add(Dropout(0.5))
    # 78 x 43
    model.add(Conv2D(128, (3, 3)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    # 76x41
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Conv2D(1024, (3, 3), padding="same"))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Conv2D(1024, (3, 3), padding="same"))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    # 76 x 41
    # model.add(Conv2D(512, (3, 3), padding="same"))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.5))
    # model.add(Conv2D(1024, (3, 3), padding="same"))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.5))
    # model.add(Conv2D(512, (3, 3), padding="same"))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.5))
    model.add(Conv2D(1024, (3, 5)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))

    # 72 x 39

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Conv2D(512, (3, 5)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    # 68 x 37
    model.add(Conv2D(512, (2, 5)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    """
    DENSE
    """
    # model.add(Dense(512))
    # model.add(LeakyReLU(alpha=0))

    # model.add(Dropout(0.5))
    # model.add(LeakyReLU(alpha=0))
    # model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.5))
    # model.add(Dense(64))
    # model.add(LeakyReLU(alpha=0))
    # model.add(Dropout(0.5))
    # model.add(Dense(32))
    # model.add(LeakyReLU(alpha=0))
    # model.add(Dropout(0.5))
    model.add(Dense(classes, activation="sigmoid"))
    # 64 x 36
    model.compile(optimizer=Adam(0.1), loss=mean_absolute_percentage_error, metrics=['accuracy'])
    model.summary()

    model.fit(train_gen, validation_data=val_gen, epochs=epochs,
              verbose=1, callbacks=[PyDriveCallback()])
    model.summary()
    model.save('/content/drive/My Drive/model.h5')


# trainsmall()

divider = DataDivider("chest_xray_data_set/metadata/chest_xray_metadata.csv")
divider.divide_data()
