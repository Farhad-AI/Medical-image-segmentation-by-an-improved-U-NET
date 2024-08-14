import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.models import *
from tensorflow.keras import metrics
from tensorflow.keras.layers import ReLU, Conv2D,BatchNormalization, MaxPooling2D, Concatenate, Conv2DTranspose,  Dropout, Input, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from data import *
import visualkeras

def double_conv( inputs, filters=64):
    conv1 = Conv2D(filters, 3, strides=1, padding = 'same', kernel_initializer = 'he_normal', )(inputs)
    batch_norm1 = BatchNormalization()(conv1)
    act1 = ReLU()(batch_norm1)

    conv2 = Conv2D(filters, 3, strides=1, padding = 'same', kernel_initializer = 'he_normal')(act1)
    batch_norm2 = BatchNormalization()(conv2)
    act2 = ReLU()(batch_norm2)

    return act2

def transp( inputs, filters=64, kernel=3, strides=1):
    transp1 = Conv2DTranspose(filters, kernel_size=kernel, strides=strides, padding = 'same', kernel_initializer = 'he_normal', )(inputs)
    batch_norm1 = BatchNormalization()(transp1)
    act1 = ReLU()(batch_norm1)
    return act1


def double_transp( inputs, first_filters, second_filters):
    transp1 = Conv2DTranspose(first_filters, kernel_size=3, strides=1, padding = 'same', kernel_initializer = 'he_normal', )(inputs)
    batch_norm1 = BatchNormalization()(transp1)
    act1 = ReLU()(batch_norm1)

    transp2 = Conv2DTranspose(second_filters, kernel_size=3, strides=1, padding = 'same', kernel_initializer = 'he_normal')(act1)
    batch_norm2 = BatchNormalization()(transp2)
    act2 = ReLU()(batch_norm2)
    return act2

def unet_by_backside3(input_size = (256,256,1), num_classes=2, pretrained_weights=None):
    inputs = Input(input_size)
    conv_block1 = double_conv(inputs, filters=64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv_block1)
    bn_pool1 = BatchNormalization()(pool1)
    #backbone1 = double_conv(conv_block1, filters=1)
    backbone1 = double_transp(conv_block1, first_filters=64, second_filters=1)

    conv_block2 = double_conv(bn_pool1,  filters=128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv_block2)
    bn_pool2 = BatchNormalization()(pool2)
    #backbone2 = double_conv(conv_block2, filters=64)
    backbone2 = double_transp(conv_block2, first_filters=128, second_filters=64)

    conv_block3 = double_conv(bn_pool2, filters=256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv_block3)
    bn_pool3 = BatchNormalization()(pool3)
    #backbone3 = double_conv(conv_block3, filters=128)
    backbone3 = double_transp(conv_block3, first_filters=256, second_filters=128)

    conv_block4 = double_conv(bn_pool3, filters=512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv_block4)
    bn_pool4 = BatchNormalization()(pool4)
    #backbone4 = double_conv(conv_block4, filters=256)
    backbone4 = double_transp(conv_block4, first_filters=512, second_filters=256)

    conv_block5 = double_conv(bn_pool4, filters=1024)
    #up5 = Conv2DTranspose( 512, (2, 2), strides=2, padding="same",activation ="relu")(conv_block5)
    up5 = transp(conv_block5, 512, kernel=2, strides=2)
    bn_up5 = BatchNormalization()(up5)
    Connect_Skip6 = Concatenate()([bn_up5, conv_block4])
    conv_block6 = double_conv(Connect_Skip6, filters=512)

    connected_trasp6 = Concatenate()([conv_block6, backbone4])
    #up6 = Conv2DTranspose( 256, (2, 2), strides=2, padding="same",activation ="relu")(connected_trasp6)
    up6 = transp(connected_trasp6, 256, kernel=2, strides=2)
    bn_up6 = BatchNormalization()(up6)

    Connect_Skip7 = Concatenate()([bn_up6, conv_block3])
    conv_block7 = double_conv(Connect_Skip7, filters=256)
    connected_trasp7 = Concatenate()([conv_block7, backbone3])
    #up7 = Conv2DTranspose( 128, (2, 2), strides=2, padding="same",activation ="relu")(connected_trasp7)
    up7 = transp(connected_trasp7, 128, kernel=2, strides=2)
    bn_up7 = BatchNormalization()(up7)

    Connect_Skip8 = Concatenate()([bn_up7, conv_block2])
    conv_block8 = double_conv(Connect_Skip8, filters=128)
    connected_trasp8 = Concatenate()([conv_block8, backbone2])
    #up8 = Conv2DTranspose( 64, (2, 2), strides=2, padding="same",activation ="relu")(connected_trasp8)
    up8 = transp(connected_trasp8, 64, kernel=2, strides=2)
    bn_up8 = BatchNormalization()(up8)

    Connect_Skip9 = Concatenate()([bn_up8, conv_block1])
    conv_block9 = double_conv(Connect_Skip9, filters=64)
    output = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv_block9)
    bn_output = BatchNormalization()(output)

    connect_transp9 = Concatenate()([backbone1, bn_output])
    output_new = Conv2D(num_classes, (1, 1), activation='sigmoid')(connect_transp9)

    model = Model(inputs = inputs, outputs = output_new)
    #model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model

model = unet_by_backside3()
#visualkeras.layered_view(model).show()
#model.save_weights("my_model.weights.h5")
from tensorflow.keras.utils import  plot_model
plot_model(model, to_file='unet_model_magrande.png', show_shapes=True, show_layer_names=True)