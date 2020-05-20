import scipy.io
import numpy as np
import cv2 as cv


class Dataset:
    def __init__(self, image_shape=(128, 128)):
        self.rootdir = 'dataset'
        self.datadir = self.rootdir + '/floorNet'
        self.image_shape = image_shape
