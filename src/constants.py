import os
import pandas as pd
from glob import glob
import numpy as np
INPUT_PATH='../data'
DATA_PATH=INPUT_PATH
TRAIN_DATA = os.path.join(DATA_PATH, "train")
TRAIN_MASKS_DATA = os.path.join(DATA_PATH, "train_masks")
TEST_DATA = os.path.join(DATA_PATH, "test")
# CSV File Paths
TRAIN_MASKS_CSV_FILEPATH = os.path.join(DATA_PATH, "train_masks.csv")
METADATA_CSV_FILEPATH = os.path.join(DATA_PATH, "metadata.csv")
# Masks to Pandas
TRAIN_MASKS_CSV = pd.read_csv(TRAIN_MASKS_CSV_FILEPATH)
TRAIN_MASKS_CSV['id'] = TRAIN_MASKS_CSV['img'].apply(lambda x: x[:-7])
METADATA_CSV = pd.read_csv(METADATA_CSV_FILEPATH)
METADATA_CSV['train'] = False
METADATA_CSV['test'] = False
all_dataset_ids = np.unique(METADATA_CSV['id'])
train_dataset_ids = np.unique(TRAIN_MASKS_CSV['id'])
test_dataset_ids = np.unique(list(set(all_dataset_ids) - set(train_dataset_ids))).tolist()
mask = METADATA_CSV.id.isin(train_dataset_ids)
METADATA_CSV.loc[mask, 'train'] = 1
mask = METADATA_CSV.id.isin(test_dataset_ids)
METADATA_CSV.loc[mask, 'test'] = 1
# List with all Picture paths for Train
train_files = glob(os.path.join(TRAIN_DATA, "*.jpg"))
train_ids = [s[len(TRAIN_DATA)+1:-4] for s in train_files]

test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = [s[len(TEST_DATA)+1:-4] for s in test_files]