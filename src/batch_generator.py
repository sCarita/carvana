import cv2
from copy import copy
import constants as c
import glob
import math
import numpy as np
import os

from tqdm import tqdm
from random import shuffle
from PIL import Image

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

class CarvanaBatchGenerator():
    
    def __init__(self, dim_x=96, dim_y=96, batch_size=32, shuffle=True, mode='Train'):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode=mode
        
    def generate(self, list_of_ids):
        shuffle(list_of_ids)
        # Generates endless batches of cars
        while 1:            
            # Generate batches
            imax = int(len(list_of_ids)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_ids_temp = [k for k in list_of_ids[i*self.batch_size:(i+1)*self.batch_size]]

                # Generate data
                X, y = self.__data_generation(list_ids_temp)

                yield X, y
        
    def __get_exploration_order(self, list_ids):
        # Find exploration order
        indexes = np.arange(len(list_ids))
        print indexes
        if self.shuffle == True:
            np.random.shuffle(indexes)
            
        return indexes
            
    def __data_generation(self, batch_ids):
        batch_images = np.concatenate([ self.pre_processed_image(_id)[np.newaxis] for _id in batch_ids])
        batch_masks = np.concatenate([ self.pre_processed_mask(_id)[np.newaxis] for _id in batch_ids])

        return batch_images, batch_masks
    
    # Pre Processing Methods
    def pre_processed_image(self, id_):
        img = self._get_image_data(id_, self.mode)

        new_w, new_h = self.dim_x, self.dim_y
        img = cv2.resize(img, (new_w,new_h))
        
        img = img/255.0
        
        return img
    
    def pre_processed_mask(self, id_):
        mask = self._get_image_data(id_, self.mode+'_mask')

        new_w, new_h = self.dim_x, self.dim_y
        mask = cv2.resize(mask, (new_w,new_h))
        mask = mask/255.0
        return np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
    
    # Private Methods
    def _rle_encode(self, img):
        pixels = img.flatten()
        pixels[0] = 0
        pixels[-1] = 0
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
        runs[1::2] = runs[1::2] - runs[:-1:2]
    
        return ' '.join(str(x) for x in runs)
    
    def _rle_decode(self, mask_rle, shape):
        s = mask_rle.split()
        
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        
        return img.reshape(shape)
    
    def _get_filename(self, image_id, image_type):
        check_dir = False
        
        if "Train" == image_type:
            ext = 'jpg'
            data_path = c.TRAIN_DATA
            suffix = ''
        elif "Train_mask" in image_type:
            ext = 'gif'
            data_path = c.TRAIN_MASKS_DATA
            suffix = '_mask'
        elif "Test" in image_type:
            ext = 'jpg'
            data_path = c.TEST_DATA
            suffix = ''
        else:
            raise Exception("Image type '%s' is not recognized" % image_type)

        if check_dir and not os.path.exists(data_path):
            os.makedirs(data_path)

        return os.path.join(data_path, "{}{}.{}".format(image_id, suffix, ext))
    
    def _get_image_data(self, image_id, image_type, **kwargs):
        if 'mask' in image_type:
            img = self._get_image_data_pil(image_id, image_type, **kwargs)
        else:
            img = self._get_image_data_opencv(image_id, image_type, **kwargs)

        return img
    
    def _get_image_data_opencv(self, image_id, image_type, **kwargs):
        fname = self._get_filename(image_id, image_type)
        img = cv2.imread(fname)
        
        assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def _get_image_data_pil(self, image_id, image_type, return_exif_md=False, return_shape_only=False):
        fname = self._get_filename(image_id, image_type)
        
        try:
            img_pil = Image.open(fname)
        except Exception as e:
            assert False, "Failed to read image : %s, %s. Error message: %s" % (image_id, image_type, e)

        if return_shape_only:
            return img_pil.size[::-1] + (len(img_pil.getbands()),)

        img = np.asarray(img_pil)
        
        assert isinstance(img, np.ndarray), "Open image is not an ndarray. Image id/type : %s, %s" % (image_id, image_type)
        
        if not return_exif_md:
            return img
        else:
            return img, img_pil._getexif()
