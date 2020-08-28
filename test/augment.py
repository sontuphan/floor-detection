import tensorflow as tf
import cv2 as cv
from random import random

image_path = tf.keras.utils.get_file(
    "cat.jpg", "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg")


def rand(min_val, max_val):
    return min_val + random()*(max_val-min_val)


def randomize():
    img = cv.imread(image_path)
    img = img/255
    (h, w, _) = img.shape

    while True:
        new_img = img
        new_img = tf.image.random_brightness(new_img, 0.2)
        new_img = tf.image.random_contrast(new_img, 0.5, 1)
        new_img = tf.image.random_saturation(new_img, 0.75, 2)
        new_img = tf.image.random_hue(new_img, 0.05)
        if tf.random.uniform(()) > 0.5:
            new_img = tf.image.flip_left_right(new_img)
        central_fraction = rand(0.8, 1)
        new_img = tf.image.central_crop(
            new_img, central_fraction=central_fraction)
        new_img = tf.image.resize(
            new_img, (h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        cv.imshow('Origin', img)
        cv.moveWindow('Origin', 100, 100)
        cv.imshow('Augmentation', new_img.numpy())
        cv.moveWindow('Augmentation', 500, 100)
        if cv.waitKey(200) & 0xFF == ord('q'):
            break
