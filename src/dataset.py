import pathlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class Dataset:
    def __init__(self, image_shape=(128, 128), batch_size=64):
        # Paths
        self.rootdir = 'dataset'
        self.datadir = self.rootdir + '/floorNet'
        self.training_set = pathlib.Path(self.datadir + '/training')
        self.validation_set = pathlib.Path(self.datadir + '/validation')
        # Params
        self.image_shape = image_shape
        self.batch_size = batch_size
        # Summary data
        self.num_training = 0
        self.num_validation = 0
        self.__calculate_data_num()

    def __calculate_data_num(self):
        self.num_training = int(len(list(self.training_set.glob('*')))/2)
        self.num_validation = int(len(list(self.validation_set.glob('*')))/2)

    def __load_img(self, path, mode='rgb'):
        img = cv.imread(path)
        if mode == 'rgb':
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        elif mode == 'gray':
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            pass
        img = cv.resize(img, self.image_shape)
        return img/255

    def print_dataset_info(self):
        print('*** Number of training data:', self.num_training)
        print('*** Number of validation data:', self.num_validation)

    def view_samples(self, samples):
        samples = list(samples)
        length = len(samples)
        plt.figure(figsize=(5, 5*length))
        for i, (raw_img, mask_img) in enumerate(samples):
            plt.subplot(length, 2, 2*i+1)
            plt.imshow(raw_img)
            plt.subplot(length, 2, 2*i+2)
            plt.imshow(mask_img)
        plt.show()

    def load_pair(self, name, mode='training'):
        raw_path = self.datadir+'/'+mode+'/'+str(name)+'.jpg'
        raw_img = self.__load_img(raw_path, 'rgb')
        mask_path = self.datadir+'/'+mode+'/'+str(name)+'_seg.jpg'
        mask_img = self.__load_img(mask_path, 'gray')
        mask_img = np.reshape(mask_img, (mask_img.shape + (1,)))
        return raw_img, mask_img

    def generator(self):
        for i in range(self.num_training):
            raw_img, mask_img = self.load_pair(str(i), 'training')
            yield raw_img, mask_img

    def prepare_for_training(self, ds):
        ds = ds.cache()
        ds = ds.shuffle(1024)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def pipeline(self):
        ds = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=((self.image_shape+(3,)), (self.image_shape+(1,)))
        )
        pipeline = self.prepare_for_training(ds)
        return pipeline
