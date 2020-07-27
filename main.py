import sys
import os
import tensorflow as tf
from test import dataset, detector

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Bugfix
# https://github.com/tensorflow/tensorflow/issues/36510
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

if __name__ == "__main__":

    if sys.argv[1] == '--dataset':
        if sys.argv[2] == 'mining':
            dataset.mining()
        if sys.argv[2] == 'show_info':
            dataset.show_info()
        if sys.argv[2] == 'view_samples':
            dataset.view_samples()

    elif sys.argv[1] == '--detector':
        if sys.argv[2] == 'mobilenet':
            detector.mobilenet()
        if sys.argv[2] == 'summary':
            detector.summary()
        if sys.argv[2] == 'show_predictions':
            detector.show_predictions()
        if sys.argv[2] == 'train':
            detector.train(True)
        if sys.argv[2] == 'predict':
            detector.predict()
        if sys.argv[2] == 'convert':
            detector.convert()
        if sys.argv[2] == 'infer':
            detector.infer()

    else:
        print("Error: Invalid option!")
