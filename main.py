import sys
import os
import tensorflow as tf
from test import dataset, detector

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Bugfixes
# https://github.com/tensorflow/tensorflow/issues/36510
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
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
