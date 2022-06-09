"""
code written by @swati sinha, @maitry sinha, @bibekananda sinha
"""
from data import *
size = 256
root_dir = './car_data'
batch_size = 4


def data_dir():
    return root_dir


def batch():
    return batch_size


def img_size():
    return size


""" 
un-comment below to run one by one...
"""

show_mask(dir=root_dir)
find_mask_labels(dir=root_dir)
make_directories(dir=root_dir)

"""
before running "make_mask_data" we must have to modify required pixel values 
of the particular mask-data on "make_mask_data" function inside data.py file..
"""

# make_mask_data(dir=root_dir, size=size)
# mask_labels_after_process(dir=root_dir)
# show_mask_after_process(dir=root_dir)
# split_data(dir=root_dir)
# make_train_data(dir=root_dir)
# make_validation_data(dir=root_dir)
