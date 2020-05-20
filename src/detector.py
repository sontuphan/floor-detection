import tensorflow as tf
from tensorflow import keras
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt

OUTPUT_CHANNELS = 2


class Detector:
    def __init__(self, image_shape=(128, 128)):
        self.image_shape = image_shape
        # Supportive stacks
        self.base_model = keras.applications.MobileNetV2(
            input_shape=(self.image_shape+(3,)), include_top=False)
        self.down_stack = self.gen_down_stack()
        self.up_stack = self.gen_up_stack()
        # Main model
        self.model = self.unet_model(OUTPUT_CHANNELS)
        self.optimizer = 'adam'
        self.loss_metric = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_metric, metrics=['accuracy'])

    def gen_down_stack(self):
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        layers = [self.base_model.get_layer(
            name).output for name in layer_names]
        down_stack = keras.Model(
            inputs=self.base_model.input, outputs=layers)
        down_stack.trainable = False
        return down_stack

    def gen_up_stack(self):
        up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]
        return up_stack

    def unet_model(self, output_channels):
        inputs = keras.layers.Input(shape=(self.image_shape+(3,)))
        x = inputs
        # Downsampling through the model
        skips = self.down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = keras.layers.Concatenate()
            x = concat([x, skip])
        # This is the last layer of the model
        last = keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2,
            padding='same')  # 64x64 -> 128x128
        x = last(x)
        return keras.Model(inputs=inputs, outputs=x)

    def train(self, ds, epochs, steps_per_epoch):

        model_history = self.model.fit(
            ds, epochs=epochs, steps_per_epoch=steps_per_epoch)

        self.model.save('models')

        loss = model_history.history['loss']

        range_of_epochs = range(epochs)

        plt.figure()
        plt.plot(range_of_epochs, loss, 'r', label='Training loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()

    def predict(self, image_batch):
        return self.model.predict(image_batch)
