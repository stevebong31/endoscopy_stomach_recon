
import numpy as np
import tensorflow as tf

import cv2, os, glob, random, argparse
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import imgaug.augmenters as iaa
import imgaug as ia

from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation, add, concatenate

# parsing args 
parser = argparse.ArgumentParser(description= 'Unet_training')

parser.add_argument('--random_seed', required = False, default = 5198)
parser.add_argument('--input_h', required = False, default = 256)
parser.add_argument('--input_w', required = False, default = 256)
parser.add_argument('--input_c', required = False, default = 3)
parser.add_argument('--data_path', required = True, default = 'data')
parser.add_argument('--batch_size', required = False, default = 128)
parser.add_argument('--file_save_name', required = False, default = 'save_model')
parser.add_argument('--epochs', required = False, default = 200)
parser.add_argument('--validation_steps', required = False, default = 16)
parser.add_argument('--mse_w', required = False, default = 1)
parser.add_argument('--ssim_w', required = False, default = 100)
parser.add_argument('--lr', required = False, default = 0.0002)

args = parser.parse_args()

ramdom_seed = args.random_seed
input_h = args.input_h
input_w = args.input_w
input_c = args.input_c
data_path = args.data_path
batch_size = args.batch_size
file_save_name = args.file_save_name
epochs = args.epochs
validation_steps = args.validation_steps
mse_w = args.mse_w
ssim_w = args.ssim_w
lr = args.lr
# random seed and gpu allow

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

tf.random.set_seed(ramdom_seed)
np.random.seed(ramdom_seed)
random.seed(ramdom_seed)
os.environ['PYTHONHASHSEED'] = str(ramdom_seed)

# loss 

def mse_dssim(y_true, y_pred):
    return mse_w * tf.losses.mean_squared_error(y_true, y_pred) + ssim_w *(1 - tf.image.ssim(y_true, y_pred, max_val = 1))

# model 

def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    return main_path


def build_network(inputs):
    to_decoder = encoder(inputs)

    path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])

    path = decoder(path, from_encoder=to_decoder)

    path = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)

    return Model(inputs=inputs, outputs=path)

model = build_network(Input(shape = (input_h, input_w, input_c)))

model.compile(optimizer = keras.optimizers.Adam(lr = lr), loss = mse_dssim)

# data load

train_x = np.load(os.path.join(data_path, 'train_x.npy'))
train_y = np.load(os.path.join(data_path, 'train_y.npy'))
test_x = np.load(os.path.join(data_path, 'test_x.npy'))
test_y = np.load(os.path.join(data_path, 'test_y.npy'))
val_x = np.load(os.path.join(data_path, 'val_x.npy'))
val_y = np.load(os.path.join(data_path, 'val_y.npy'))

def image_aug_batch(img, label):
    # images = h w c
    # label bbox = (left, top, right, bottom)
    seq = iaa.Sequential([
    iaa.Multiply((0.9, 1.1)),
    iaa.Affine(
        translate_px={"x": 10, "y": 10},
        scale=(0.95, 1.05),
        rotate=(-180, 180)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5)
    ])
    images_aug, labels_aug = seq(images = img.astype(np.float32)/255. ,heatmaps = label.astype(np.float32)/255.)
    return images_aug, labels_aug

def data_generator(train_imgs, train_labels, batch_size, aug = True):
    idx = 0
    while 1:
        idx_list = list(range(0,len(train_imgs)))
        random.shuffle(idx_list)
        if idx > len(idx_list) - batch_size:
            tmp_list = idx_list[idx:]
            idx = 0
        else:
            tmp_list = idx_list[idx:idx + batch_size]
            idx = idx + batch_size
        batch_images = train_imgs[tmp_list]
        batch_labels = train_labels[tmp_list]
        
        if aug == True:
            batch_images, batch_labels = image_aug_batch(batch_images, batch_labels)
        else:
            batch_images, batch_labels = batch_images / 255., batch_labels / 255.
        yield batch_images, batch_labels
        
train_gen = data_generator(train_x, train_y, batch_size, aug = True)
test_gen = data_generator(test_x, test_y, batch_size, aug = False)
val_gen = data_generator(val_x, val_y, batch_size, aug = False)

print('epochs :', int(len(train_x)/batch_size))
print('val step epochs :', int(len(val_x)/batch_size))

try:
    if not os.path.exists(file_save_name):
        os.makedirs(file_save_name)
except OSError:
    print ('Error: Creating directory. ' +  file_save_name)
    
filepath = f'{file_save_name}' + '/model-{epoch:04d}-{validation_loss:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, save_weights_only=True)

model_hist = model.fit(train_gen, epochs = epochs, steps_per_epoch =  int(len(train_x)/batch_size),
                    callbacks = [checkpoint], 
                    validation_data = val_gen, validation_steps = validation_steps, use_multiprocessing=True)
