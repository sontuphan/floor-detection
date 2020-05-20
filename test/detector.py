import os
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from src.detector import Detector
from src.dataset import Dataset


def summary():
    detector = Detector(image_shape=(128, 128))
    tf.keras.utils.plot_model(detector.model, show_shapes=True)
    img = cv.imread('model.png')
    img = cv.resize(img, (512, 768))
    cv.imshow('Model summary', img)
    cv.waitKey()
    os.remove('model.png')


def __display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def __create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions():
    detector = Detector(image_shape=(128, 128))
    ds = Dataset(image_shape=(128, 128))
    pipeline = ds.pipeline()
    for image, mask in pipeline.take(1):
        pred_mask = detector.predict(image)
        __display([image[0], mask[0], __create_mask(pred_mask)])


def train():
    image_shape = (128, 128)
    batch_size = 64
    detector = Detector(image_shape)
    ds = Dataset(image_shape, batch_size)
    pipeline = ds.pipeline()
    epochs = 20
    steps_per_epoch = ds.num_training//batch_size
    detector.train(pipeline, epochs, steps_per_epoch)
