import pathlib
import cv2 as cv
import numpy as np
import tensorflow as tf
from random import random


class Dataset:
    def __init__(self, image_shape=(128, 128), batch_size=64):
        # Paths
        self.rootdir = 'dataset'
        self.datadir = self.rootdir + '/FLOORNET'
        self.training_set = pathlib.Path(self.datadir + '/training')
        self.validation_set = pathlib.Path(self.datadir + '/validation')
        # Params
        self.image_shape = image_shape
        self.batch_size = batch_size
        # Summary data
        self.num_training, self.num_validation = self._calculate_data_num()

    def _rand(self, min_val, max_val):
        return min_val + random()*(max_val-min_val)

    def _resize(self, img, shape):
        return tf.image.resize(img, shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def _calculate_data_num(self):
        num_training = int(len(list(self.training_set.glob('*')))/2)
        num_validation = int(len(list(self.validation_set.glob('*')))/2)
        return num_training, num_validation

    def _load_img(self, path, mode='rgb'):
        img = cv.imread(path)
        if mode == 'rgb':
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if mode == 'gray':
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, self.image_shape, interpolation=cv.INTER_NEAREST)
        return img/255

    @tf.function
    def _augment(self, img, mask):
        img = tf.cast(img, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        # Mask-effected augmentations
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        central_fraction = self._rand(0.8, 1)
        img = tf.image.central_crop(img, central_fraction=central_fraction)
        mask = tf.image.central_crop(mask, central_fraction=central_fraction)
        # Mask-free augmentations
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.5, 1)
        img = tf.image.random_saturation(img, 0.75, 2)
        img = tf.image.random_hue(img, 0.05)
        # Normalize the image shape
        img = self._resize(img, self.image_shape)
        img = (img - 0.5) / 0.5  # MobileNetv2 [-1,1]
        mask = self._resize(mask, self.image_shape)
        return img, mask

    def print_dataset_info(self):
        print('*** Number of training data:', self.num_training)
        print('*** Number of validation data:', self.num_validation)

    def _load_pair(self, name, mode='training'):
        raw_path = self.datadir+'/'+mode+'/'+name+'.jpg'
        raw_img = self._load_img(raw_path, 'rgb')
        mask_path = self.datadir+'/'+mode+'/'+name+'_seg.jpg'
        mask_img = self._load_img(mask_path, 'gray')
        mask_img = np.reshape(mask_img, (mask_img.shape + (1,)))
        return raw_img, mask_img

    def generator(self, mode):
        mode = mode.decode('ascii')
        names = self.num_training if mode == 'training' else self.num_validation
        for i in range(names):
            raw_img, raw_mask = self._load_pair(str(i), mode)
            normalized_img, normalized_mask = self._augment(raw_img, raw_mask)
            yield normalized_img, normalized_mask

    def prepare_ds(self, ds, mode='training'):
        if mode == 'training':
            ds = ds.cache()
            ds = ds.repeat()
            ds = ds.shuffle(1024)
            ds = ds.batch(self.batch_size)
            ds = ds.map(self._augment,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        if mode == 'validation':
            ds = ds.batch(self.batch_size)
        return ds

    def pipeline(self):
        # Training dataset
        training_ds = tf.data.Dataset.from_generator(
            self.generator,
            args=['training'],
            output_types=(tf.float32, tf.float32),
            output_shapes=((self.image_shape+(3,)), (self.image_shape+(1,)))
        )
        training_pipeline = self.prepare_ds(training_ds, 'training')
        # validation dataset
        validation_ds = tf.data.Dataset.from_generator(
            self.generator,
            args=['validation'],
            output_types=(tf.float32, tf.float32),
            output_shapes=((self.image_shape+(3,)), (self.image_shape+(1,)))
        )
        validation_pipeline = self.prepare_ds(validation_ds, 'validation')
        return training_pipeline, validation_pipeline
