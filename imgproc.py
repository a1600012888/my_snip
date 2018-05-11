#!/usr/bin/env mdl

import numpy as np
import os
import cv2
import skimage.io as io
import skimage.transform as transform
def resize_ensure_shortest_edge(img, edge_length, return_ratio=False):
    assert isinstance(edge_length, int) and edge_length > 0, edge_length
    h, w = img.shape[:2]
    if h < w:
        ratio = float(edge_length) / h
        th, tw = edge_length, max(1, int(ratio*w))
    else:
        ratio = float(edge_length) / w
        th, tw = max(1, int(ratio*h)), edge_length

    ret = cv2.resize(img, (tw, th))
    if return_ratio:
        return ret, ratio
    return ret

def _clip_normalize(img):
    return np.clip(img, 0, 255).astype('uint8')



def gaussian_noise(img, sigma = 2):

    img = img + sigma * np.random.normal(size = img.shape)
    img = _clip_normalize(img)

    return img

class normalize_and_aug(object):
## for augs
    sigma = 2
    flip_prob = 0.5
    color_jitter = (0.1, 0.1, 0.1)  # brightness, contrast, saturate

    ## for normalize(resize and crop)
    short_len = 256
    shape = [224, 224]
    crop_mode_all = ['center', 'random'] ## 0---1
    crop_mode = 1

    doaug = True

    def __init__(self, doaug = False):
        self.doaug = doaug

    def __call__(self, img):
        '''
        expecting imgs of shape[x, y, 3] as np.array
        '''
        assert img.shape[-1] == 3
        assert img.ndim == 3

        img = self.normalize(img)

        if self.doaug:
            img = self.aug(img)

        return img
    def set_aug(self, sigma = 0, flip_prob = 0, color_jitter = (0., 0., 0.)):

        self.sigma = sigma
        self.flip_prob = flip_prob
        self.color_jitter = color_jitter

    def rescale(self, img):

        img = np.array(resize_ensure_shortest_edge(img, self.short_len))


        return img
    def crop(self, img):
        if self.crop_mode_all[self.crop_mode] == 'center':

            img = center_crop(img, self.shape)
        if self.crop_mode_all[self.crop_mode] == 'random':
            img = random_crop(img, self.shape)
        return img

    def normalize(self, img):

        assert img.shape[-1] == 3
        assert img.ndim == 3
        img = self.rescale(img)
        img = self.crop(img)
        img = np.clip(img, 0, 255)
        #assert img.shape[:2] == self.shape
        #print(img.shape)
        return img
    def horizontal_flip(self, img, prob = None):
        '''
        img should of nidm == 3
        prob should be of [0, 1]
        '''
        assert img.ndim == 3


        if prob == None:
            prob = self.flip_prob
        assert prob >= float(0) and prob <= float(1)
        if np.random.rand() < prob:
            img = img[:, ::-1]
        return img

    def add_gaussian(self, img):
        noise = self.sigma * np.random.normal(0, 1, img.shape)
        n_img = np.clip(img + noise, 0, 255)

        return n_img

    def aug(self, img):
        assert img.shape[-1] == 3
        assert img.ndim == 3

        img = self.horizontal_flip(img)

        img = self.add_gaussian(img)

        img = color_jitter(img, *self.color_jitter)

        img = np.clip(img, 0, 255)
        return img



def grayscale(img):
    '''
    img should be RGB
    of shape [x,y, 3]

    return:  gray of shape (x, y ,1)
    '''
    #assert img.shape == (224, 224, 3), img.shape
    w = np.array([0.299, 0.587, 0.114]).reshape(1, 1, 3)

    #print(w.shape)
    gs = np.zeros(img.shape[:2])
    gs = (img * w).sum(axis = 2, keepdims = True)

    #print(gs.shape)
    return gs
def brightness_aug(img, val):
    #assert img.shape == (224, 224, 3), img.shape
    '''
    (1-val, 1+val)
    '''
    alpha = 1. + val * (np.random.rand() * 2 - 1)

    img = img * alpha
    return img

def contrast_aug(img, val):
   # assert img.shape == (224, 224, 3), img.shape
    alpha = 1. + val * (np.random.rand() * 2 - 1)
    gs = grayscale(img)
    gs[:] = gs.mean()

    img = img * alpha + gs * (1 - alpha)

    return img
def saturation_aug(img, val):
    #assert img.shape == (224, 224, 3), img.shape
    alpha = 1. + val * (np.random.rand() * 2 - 1)
    gs = grayscale(img)

    img = img * alpha + gs * (1 - alpha)

    return img
def color_jitter(img, brightness = 0.2, contrast = 0.2, saturation = 0.2):
    '''
    img should be in shape of [x, y, 3]
    '''

    augs = [
        (brightness_aug, brightness),
        (contrast_aug, contrast),
        (saturation_aug, saturation)
    ]
    np.random.shuffle(augs)

    for aug, val in augs:
        img = aug(img, val)

    return img


# In[232]:


def center_crop(img, shape):
    '''
    img of shape[x, y, 3]  HWC w-y h-x
    '''

    h, w = img.shape[:2]

    assert h >= shape[0] and w >= shape[1]

    sx, sy = (w-shape[1])//2, (h-shape[0])//2
    img = img[sy:sy+shape[0], sx:sx+shape[1]]

    return img

def random_crop(img, shape):
    '''
    img of shape[x, y, 3]  HWC
    '''

    h, w = img.shape[:2]

    assert h >= shape[0] and w >= shape[1]

    top = np.random.randint(0, h - shape[0])
    left = np.random.randint(0, w - shape[1])

    #print(h, w, top, left)
    img = img[top:top + shape[0], left:left + shape[1]]


    return img

def rotate(image, angle):

    rotate_angel = (random.random() / 180 * np.pi) * angel
	# Create Afine transform
	afine_tf = transform.AffineTransform(rotation=rotate_angel)

	# Apply transform to image data
	image = transform.warp(image, inverse_map=afine_tf, mode='edge')

	return image
# vim: ts=4 sw=4 sts=4 expandtab
