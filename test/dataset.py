from src.factory import Factory


def mining():
    factory = Factory(image_shape = (128, 128))
    factory.mining('training')
    factory.mining('validation')
