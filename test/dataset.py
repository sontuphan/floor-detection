from src.factory import Factory
from src.dataset import Dataset


def mining():
    factory = Factory(image_shape=(224, 224))
    factory.mining('training')
    factory.mining('validation')


def show_info():
    ds = Dataset(image_shape=(224, 224))
    ds.print_dataset_info()


def view_samples():
    ds = Dataset(image_shape=(224, 224))
    pl = ds.pipeline()
    num_of_samples = 5
    for raw_imgs, mask_imgs in pl.take(1):
        samples = zip(raw_imgs[:num_of_samples], mask_imgs[:num_of_samples])
        ds.view_samples(samples)
