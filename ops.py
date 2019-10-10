from scipy import misc
import collections
import tensorflow as tf
import numpy as np
import math
import time
import random
import glob
import os
import fnmatch
import ntpath
import pickle

SKIP_ARGS = ['learning_rate', 'batch_size', 'epochs', 'data_dir', 'save_freq']

def createCheckpoint(args):

    checkpoint_dir = ''
    info = {}
    for k in vars(args):
        info[k] = getattr(args, k)
        if k not in SKIP_ARGS:
            checkpoint_dir += str(getattr(args,k))+'_'

    checkpoint_dir = os.path.join('checkpoints', checkpoint_dir[:-1])
    images_dir = os.path.join(checkpoint_dir, 'images')

    os.makedirs(images_dir, exist_ok=True)

    with open(os.path.join(checkpoint_dir, 'info.pkl'), 'wb') as f:
        pickle.dump(info, f)

    return checkpoint_dir
   

def normalize(image):
    return (image / 127.5) - 1.0


def unnormalize(img):
    img = (img + 1.)
    img *= 127.5
    return img


def deprocess(x):
    """ [-1,1] -> [0, 255] """
    return (x + 1.0) * 127.5


def preprocess(x):
    """ [0,255] -> [-1, 1] """
    return (x / 127.5) - 1.0


def getPaths(data_dir, ext='png'):
    """ Returns a list of paths """
    pattern = '*.' + ext
    image_paths = []
    for d, s, fList in os.walk(data_dir):
        for filename in fList:
            if fnmatch.fnmatch(filename, pattern):
                fname_ = os.path.join(d, filename)
                image_paths.append(fname_)
    return image_paths
