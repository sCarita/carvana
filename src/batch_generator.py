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

"""
# random example images
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
            sometimes(iaa.Affine(
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
            ],
            random_order=True
        )
    ],
    random_order=True
)
"""
class CarvanaBatchGenerator():
    
    def __init__(self, mode, batch_size, dims_area=None, val_split=0.2):
        self.batch_size = batch_size
        self.dims_area = dims_area or (96,96)

        if mode is "train":
            # Create Train/Val Split
            self.val_split = 1 - val_split
            self.split_index = int(len(c.train_ids)*self.val_split)
            self.data = copy(c.train_ids)
            # Get Training and Validation IDs
            self.shuffle_train_val_ids()
            self.mode = 'Train'
        elif mode is "test":
            self.image_ids = c.test_ids
            self.n_images = len(c.test_ids)
            self.mode = 'Test'
        else:
            raise TypeError
            
    def shuffle_train_val_ids(self):
        shuffle(self.data)

        self.image_ids = self.data[:self.split_index]
        self.val_image_ids = self.data[self.split_index:]
            
    def batches_per_epoch(self):
        return int(len(self.image_ids) / self.batch_size)
    
    def val_batches_per_epoch(self):
        return int(len(self.val_image_ids) / self.batch_size)
        
    def get_batches(self):
        
        if self.mode is 'Train':
            self.shuffle_train_val_ids()
            for batch_ids in np.array_split(self.image_ids, self.batches_per_epoch()):
                yield self._get_tuple_batch_train(batch_ids)
        else:
            shuffle(self.image_ids)
            for batch_ids in np.array_split(self.image_ids, self.batches_per_epoch()):
                yield self._get_tuple_batch_test(batch_ids)
                
    def get_val_batches(self):
        shuffle(self.val_image_ids)
    
        for batch_ids in np.array_split(self.val_image_ids, self.val_batches_per_epoch()):
            yield self._get_tuple_batch_train(batch_ids)

    def _get_tuple_batch_train(self, batch_ids):
        batch_images = np.concatenate([ self.pre_processed_image(_id)[np.newaxis] for _id in batch_ids])
        # aug_batch_images = seq.augment_images(batch_images)
        batch_masks = np.concatenate([ self.pre_processed_mask(_id)[np.newaxis] for _id in batch_ids])

        return batch_images, batch_masks
    
    def _get_tuple_batch_test(self, batch_ids):
        batch_images = np.concatenate([ self.pre_processed_image(_id)[np.newaxis] for _id in batch_ids])
        # aug_batch_images = seq.augment_images(batch_images)
        
        return batch_images
    
    # Pre Processing Methods
    def pre_processed_image(self, id_):
        img = self._get_image_data(id_, self.mode)
        #TARGET_PIXEL_AREA = self.dims_area[0]*self.dims_area[1]

        #ratio = float(img.shape[1]) / float(img.shape[0])
        #new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
        #new_w = int((new_h * ratio) + 0.5)
        new_w, new_h = self.dims_area
        img = cv2.resize(img, (new_w,new_h))
        
        img = img/255.0
        
        return img
    
    def pre_processed_mask(self, id_):
        mask = self._get_image_data(id_, self.mode+'_mask')
        #TARGET_PIXEL_AREA = self.dims_area[0]*self.dims_area[1]

        #ratio = float(mask.shape[1]) / float(mask.shape[0])
        #new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
        #new_w = int((new_h * ratio) + 0.5)
        new_w, new_h = self.dims_area
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
