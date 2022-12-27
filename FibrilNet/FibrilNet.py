# =========================================================================
#   (c) Copyright 2021
#   All rights reserved
#   Programs written by Haodi Jiang
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================


import warnings
import os
import sys
warnings.filterwarnings('ignore')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    # print('turning logging of is not available')
    pass
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import cv2
import matplotlib
import skimage.io as io
import skimage.transform as trans
from astropy.io import fits
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
sys.stderr = stderr
matplotlib.use('TkAgg')
from keras import backend as K


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == "tensorflow":
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                inputs, ksize=ksize, strides=strides, padding=padding
            )
        else:
            errmsg = "{} backend is not supported for layer {}".format(
                K.backend(), type(self).__name__
            )
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, "int32")
            input_shape = K.tf.shape(updates, out_type="int32")
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.size[0],
                    input_shape[2] * self.size[1],
                    input_shape[3],
                )
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype="int32")
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = K.reshape(
                K.tf.range(output_shape[0], dtype="int32"), shape=batch_shape
            )
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype="int32")
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3],
        )


def FibrilNet(pretrained_weights=None,input_shape=(720, 720, 1), n_labels=2, kernel=3, pool_size=(2, 2), output_mode="sigmod"):
    inputs = Input(shape=input_shape)

    # encoder
    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = Activation("relu")(conv_1)
    conv_1 = BatchNormalization()(conv_1)

    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = Activation("relu")(conv_2)
    conv_2 = BatchNormalization()(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)
    pool_1 = Dropout(0.5)(pool_1, training=True)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = Activation("relu")(conv_3)
    conv_3 = BatchNormalization()(conv_3)

    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = Activation("relu")(conv_4)
    conv_4 = BatchNormalization()(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)
    pool_2 = Dropout(0.5)(pool_2, training=True)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = Activation("relu")(conv_5)
    conv_5 = BatchNormalization()(conv_5)

    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = Activation("relu")(conv_6)
    conv_6 = BatchNormalization()(conv_6)

    # conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    # conv_7 = Activation("relu")(conv_7)
    # conv_7 = BatchNormalization()(conv_7)
    conv_7 = conv_6

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)
    pool_3 = Dropout(0.5)(pool_3, training=True)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = Activation("relu")(conv_8)
    conv_8 = BatchNormalization()(conv_8)

    conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = Activation("relu")(conv_9)
    conv_9 = BatchNormalization()(conv_9)

    # conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    # conv_10 = Activation("relu")(conv_10)
    # conv_10 = BatchNormalization()(conv_10)
    conv_10 = conv_9

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)
    pool_4 = Dropout(0.5)(pool_4, training=True)

    # conv_11 = Convolution2D(1024, (kernel, kernel), padding="same")(pool_4)
    # conv_11 = Activation("relu")(conv_11)
    # conv_11 = BatchNormalization()(conv_11)
    #
    # conv_12 = Convolution2D(1024, (kernel, kernel), padding="same")(conv_11)
    # conv_12 = Activation("relu")(conv_12)
    # conv_12 = BatchNormalization()(conv_12)
    #
    # # conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    # # conv_13 = Activation("relu")(conv_13)
    # # conv_13 = BatchNormalization()(conv_13)
    # conv_13 = conv_12

    # pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    # # print("Build enceder done..")
    #
    ## between encoder and decoder
    conv_14 = Convolution2D(1024, (kernel, kernel), padding="same")(pool_4)
    conv_14 = Activation("relu")(conv_14)
    conv_14 = BatchNormalization()(conv_14)

    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = Activation("relu")(conv_15)
    conv_15 = BatchNormalization()(conv_15)
    #
    # # conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    # # conv_16 = Activation("relu")(conv_16)
    # # conv_16 = BatchNormalization()(conv_16)
    conv_16 = conv_15

    # decoder
    unpool_1 = MaxUnpooling2D(pool_size)([conv_16, mask_4])
    concat_1 = Concatenate()([unpool_1, conv_10])
    concat_1 = Dropout(0.5)(concat_1, training=True)

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(concat_1)
    conv_17 = Activation("relu")(conv_17)
    conv_17 = BatchNormalization()(conv_17)

    conv_18 = Convolution2D(256, (kernel, kernel), padding="same")(conv_17)
    conv_18 = Activation("relu")(conv_18)
    conv_18 = BatchNormalization()(conv_18)

    # conv_19 = Convolution2D(512, (kernel, kernel), padding="same")(conv_18)
    # conv_19 = Activation("relu")(conv_19)
    # conv_19 = BatchNormalization()(conv_19)
    conv_19 = conv_18

    unpool_2 = MaxUnpooling2D(pool_size)([conv_19, mask_3])
    concat_2 = Concatenate()([unpool_2, conv_7])
    concat_2 = Dropout(0.5)(concat_2, training=True)

    conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(concat_2)
    conv_20 = Activation("relu")(conv_20)
    conv_20 = BatchNormalization()(conv_20)

    conv_21 = Convolution2D(128, (kernel, kernel), padding="same")(conv_20)
    conv_21 = Activation("relu")(conv_21)
    conv_21 = BatchNormalization()(conv_21)

    # conv_22 = Convolution2D(256, (kernel, kernel), padding="same")(conv_21)
    # conv_22 = Activation("relu")(conv_22)
    # conv_22 = BatchNormalization()(conv_22)
    conv_22 = conv_21

    unpool_3 = MaxUnpooling2D(pool_size)([conv_22, mask_2])
    concat_3 = Concatenate()([unpool_3, conv_4])
    concat_3 = Dropout(0.5)(concat_3, training=True)

    conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(concat_3)
    conv_23 = Activation("relu")(conv_23)
    conv_23 = BatchNormalization()(conv_23)

    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = Activation("relu")(conv_24)
    conv_24 = BatchNormalization()(conv_24)

    # conv_25 = Convolution2D(128, (kernel, kernel), padding="same")(conv_24)
    # conv_25 = Activation("relu")(conv_25)
    # conv_25 = BatchNormalization()(conv_25)
    conv_25 = conv_24


    unpool_4 = MaxUnpooling2D(pool_size)([conv_25, mask_1])
    concat_4 = Concatenate()([unpool_4, conv_2])
    concat_4 = Dropout(0.5)(concat_4, training=True)

    conv_26 = Convolution2D(64, (kernel, kernel), padding="same")(concat_4)
    conv_26 = Activation("relu")(conv_26)
    conv_26 = BatchNormalization()(conv_26)

    conv_27 = Convolution2D(64, (kernel, kernel), padding="same")(conv_26)
    conv_27 = Activation("relu")(conv_27)
    conv_27 = BatchNormalization()(conv_27)


    # unpool_5 = MaxUnpooling2D(pool_size)([conv_27, mask_1])
    # concat_5 = Concatenate()([unpool_5, conv_2])
    #
    # conv_28 = Convolution2D(64, (kernel, kernel), padding="same")(concat_5)
    # conv_28 = Activation("relu")(conv_28)
    # conv_28 = BatchNormalization()(conv_28)
    #
    # conv_29 = Convolution2D(64, (kernel, kernel), padding="same")(conv_28)
    # conv_29 = Activation("relu")(conv_29)
    # conv_29 = BatchNormalization()(conv_29)

    conv_30 = Conv2D(1, 1, activation='sigmoid')(conv_27)

    model = Model(inputs=inputs, outputs=conv_30)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coefficient, precision_smooth, recall_smooth])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    # print(model.summary())
    return model


def adjust_data(img, mask):
    if np.max(img) > 1:
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask


def train_generator(batch_size, train_path, image_folder, mask_folder):

    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=(720,720),
        batch_size=batch_size,
        seed=1)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=(720,720),
        batch_size=batch_size,
        seed=1)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img,mask = adjust_data(img, mask)
        yield img, mask


def validation_generator(batch_size, train_path,image_folder, mask_folder):

    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=(720,720),
        batch_size=batch_size,
        seed=1)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=(720,720),
        batch_size=batch_size,
        seed=1)
    validation_generator = zip(image_generator, mask_generator)
    for (img, mask) in validation_generator:
        img,mask = adjust_data(img, mask)
        yield img, mask


def test_generator(test_path, target_size=(720,720)):
    for name in os.listdir(test_path):
        img = io.imread(os.path.join(test_path, name), as_gray=True)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)
        yield img


def predict(model, image, T=50):
    # predict stochastic dropout model T times
    p_hat = []
    for t in range(T):
        p_hat.append(model.predict(image)[0])
    p_hat = np.array(p_hat)

    # mean prediction
    prediction = np.mean(p_hat, axis=0)

    # estimate uncertainties
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2

    return np.squeeze(prediction), np.squeeze(aleatoric), np.squeeze(epistemic)


def save_result(save_path, result):
    for i, item in enumerate(result):
        img = np.round(item) * 255
        cv2.imwrite(os.path.join(save_path, "predicted_mask_{0:03}.png".format(i)), img)





