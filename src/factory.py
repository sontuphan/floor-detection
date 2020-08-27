import scipy.io
import numpy as np
import cv2 as cv
import pandas as pd


class Factory:
    def __init__(self):
        self.rootdir = 'dataset'
        self.datadir = self.rootdir + '/ADE20K_2016_07_26'
        self.outdir = self.rootdir + '/ade20k'
        self.matfile = self.datadir+'/index_ade20k.mat'

    def tranfer_image(self, img):
        # (BGR)=(*,208,30) is floor
        mask = np.zeros(img.shape) + [255, 255, 255]
        mask[(img[:, :, 1] == 208) & (img[:, :, 2] == 30)] = [0, 0, 0]
        return mask

    def filter_floor(self, label_txt):
        df = pd.read_csv(label_txt, sep='#', header=None)
        # Column 5: label
        for [label] in df[[4]].to_numpy():
            name = label.replace(' ', '')
            if name == 'floor':
                return True
        return False

    def loader(self, mode='training'):
        mat = scipy.io.loadmat(self.matfile)
        data = mat['index']
        filename = np.squeeze(data[0][0][0])
        folder = np.squeeze(data[0][0][1])
        imgs = map(lambda fd, fn: fd[0]+'/'+fn[0], folder, filename)
        img_path = (self.rootdir+'/' +
                    img for img in imgs if mode in img.split('/'))
        # Including raw image, attribute text file, and mask image
        data_path = ((f, f.split('.')[0]+'_atr.txt',
                      f.split('.')[0]+'_seg.png') for f in img_path)
        # Get only images with floor
        data_path = filter(lambda row: self.filter_floor(row[1]), data_path)
        return data_path

    def mining(self, mode):
        data_path = self.loader(mode)
        for i, (img, txt, mask) in enumerate(data_path):
            print('======== Counter:', i)
            # Raw image
            raw_img = cv.imread(img)
            cv.imwrite(self.outdir+'/'+mode+'/'+str(i)+'.jpg', raw_img)
            # Mask image
            mask_img = cv.imread(mask)
            mask_img = self.tranfer_image(mask_img)
            cv.imwrite(self.outdir+'/'+mode+'/'+str(i)+'_seg.jpg', mask_img)
