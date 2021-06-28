# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import cv2, os, glob, random, tqdm
import numpy as np 
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import imgaug as ia


# %%
ramdom_seed = 5198
np.random.seed(ramdom_seed)
random.seed(ramdom_seed)
os.environ['PYTHONHASHSEED'] = str(ramdom_seed)


# %%
def load_data(path):
    all_data = []
    data_list = glob.glob(path + '/*.png')
    for tmp in tqdm.tqdm(data_list): 
        img = cv2.cvtColor(cv2.imread(tmp),cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256,256))
        all_data.append(img)
    all_data = np.array(all_data)
    return all_data


# %%
def image_aug_batch(img):
    #
    seq = iaa.Sequential([
    iaa.MultiplyAndAddToBrightness(mul=(0.9, 1.1), add=(-10, 10)),
    iaa.MotionBlur(k=3, angle=[-45, 45]),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(
        translate_px={"x": 10, "y": 10},
        scale=(0.90, 1.1),
        rotate=(-5, 5), mode='edge')])
    images_aug = seq(images = img)
    return images_aug


# %%
def data_loader(data1, data2, batch, aug = True):
    while 1:
        data1_batch = data1[np.random.choice(np.arange(0,len(data1)), batch)]
        data2_batch = data2[np.random.choice(np.arange(0,len(data2)), batch)]
        if aug == True:
            data1_aug = image_aug_batch(data1_batch)/127.5 - 1. 
            data2_aug = image_aug_batch(data2_batch)/127.5 - 1.
        else:
            data1_aug = data1_batch
            data2_aug = data1_batch
        yield data1_aug, data2_aug

# %% [markdown]
# 

