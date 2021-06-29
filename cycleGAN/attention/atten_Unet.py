# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, LeakyReLU, Activation, Input, add, multiply
from tensorflow.keras.layers import concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K
import instancenormalization



def up_and_concate(down_layer, layer, data_format='channels_last'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])

    return concate


def attention_up_and_concate(down_layer, layer, data_format='channels_last'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x


def att_unet(img_w, img_h, data_format='channels_last'):
    inputs = Input((img_w, img_h, 3))
    x = inputs
    depth = 4
    features = 32
    skips = []
    for i in range(depth):
        x = Conv2D(features, 3, activation=LeakyReLU(), padding='same')(x)
        x = instancenormalization.InstanceNormalization()(x)
        x = Conv2D(features, 3, activation=LeakyReLU(), padding='same')(x)
        x = instancenormalization.InstanceNormalization()(x)
        skips.append(x)
        x = MaxPooling2D(2)(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation=LeakyReLU(), padding='same', data_format=data_format)(x)
    x = instancenormalization.InstanceNormalization()(x)
    x = Conv2D(features, (3, 3), activation=LeakyReLU(), padding='same', data_format=data_format)(x)
    x = instancenormalization.InstanceNormalization()(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = Conv2D(features, (3, 3), activation=LeakyReLU(), padding='same', data_format=data_format)(x)
        x = instancenormalization.InstanceNormalization()(x)
        x = Conv2D(features, (3, 3), activation=LeakyReLU(), padding='same', data_format=data_format)(x)
        x = instancenormalization.InstanceNormalization()(x)

    conv6 = Conv2D(3, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = Activation('tanh')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    #model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
    return model



# %%



