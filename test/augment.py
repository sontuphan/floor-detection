import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt

image_path = tf.keras.utils.get_file(
    "cat.jpg", "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg")


def brightness():
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img/255

    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.5, 1)
    img = tf.image.random_saturation(img, 1, 5)

    plt.imshow(img)
    plt.show()
