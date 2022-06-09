import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import splitfolders
import tarfile
# pat = "images.tar.gz"


# ##if data in tarfile then extract it ##########################
def extract_file(pat, to_directory='.'):
    if pat.endswith('.tar.gz') or pat.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    cwd = os.getcwd()
    os.chdir(to_directory)
    try:
        file = opener(pat, mode)
        try: file.extractall()
        finally: file.close()
    finally:
        os.chdir(cwd)


def make_directories(dir):
    try:
        os.makedirs('unet_models', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        os.makedirs('{}/dir/images'.format(dir), exist_ok=True)
        os.makedirs('{}/dir/masks'.format(dir), exist_ok=True)
        os.makedirs('{}/data/training_data/train_images/train'.format(dir), exist_ok=True)
        os.makedirs('{}/data/training_data/train_masks/train'.format(dir), exist_ok=True)
        os.makedirs('{}/data/training_data/val_images/val'.format(dir), exist_ok=True)
        os.makedirs('{}/data/training_data/val_masks/val'.format(dir), exist_ok=True)
        print("All Directories are created successfully..")
    except OSError as error:
        print("Directory can not be created")


def show_mask(dir):
    path = '{}/masks/'.format(dir)
    lst = os.listdir(path)
    mask = cv2.imread(os.path.join(path, lst[3]), 0)
    # mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    plt.imshow(mask)
    plt.axis('off')
    plt.show()


def find_mask_labels(dir):
    path = '{}/masks/'.format(dir)
    mask_list = os.listdir(path)
    labels = []
    for m in mask_list:
        msk = cv2.imread(os.path.join(path, m), 0).copy()
        label = np.unique(msk)
        for l in label:
            if l not in labels:
                labels.append(l)
    labels = np.sort(labels)
    print('Labels       :', labels)
    print('No. of class :', len(labels))
    print('Shape of mask:', msk.shape)


# ########### clean mask data for fashion mask #################
def make_class_fashion(mask):
    # for fashion mask
    try:
        mask[mask >= 59] = 0
    except:pass
    return mask


# ########### clean mask data for pet mask #################
def make_class_pet(mask):
    # for fashion mask
    # msk = mask[:, :, 1].copy()
    try:
        mask[mask == 2] = 0  # background
        mask[mask == 3] = 2
    except: pass
    return mask


def make_class_hiphop(mask):
    # for fashion mask
    try:
        mask[mask <= 253] = 0  # background
        mask[mask >= 254] = 1
    except: pass
    return mask


def make_class_car(mask):
    # for car mask
    try:
        mask[mask == 255] = 1
    except: pass
    return mask


# ### put bellow the above function ########################
def make_mask_data(dir, size=256):
    images = "{}/images".format(dir)
    target_images = "{}/dir/images".format(dir)
    masks = "{}/masks".format(dir)
    target_masks = "{}/dir/masks".format(dir)
    for img, msk in zip(os.listdir(images), os.listdir(masks)):
        try:
            image = cv2.imread(os.path.join(images, img), 1)
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
            num_img = img.split('.')[0]
            cv2.imwrite(os.path.join(target_images, num_img+'.jpg'), image)
            mask = cv2.imread(os.path.join(masks, msk), 1)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            mask = make_class_car(mask)
            cv2.imwrite(os.path.join(target_masks, num_img+'.png'), mask)
        except: pass
    print('data generated...')


def mask_labels_after_process(dir):
    path = '{}/dir/masks/'.format(dir)
    mask_list = os.listdir(path)
    labels = []
    for m in mask_list:
        msk = cv2.imread(os.path.join(path, m), 0)
        label = np.unique(msk)
        for l in label:
            if l not in labels:
                labels.append(l)
    labels = np.sort(labels)
    print('Labels       :', labels)
    print('No. of class :', len(labels))
    print('Shape of mask:', msk.shape)
    return len(labels)


def show_mask_after_process(dir):
    path = '{}/dir/masks/'.format(dir)
    lst = os.listdir(path)
    for m in lst[8:10]:
        mask = cv2.imread(os.path.join(path, m), 0)
        plt.imshow(mask)
        plt.axis('off')
        plt.show()


def split_data(dir):
    splitfolders.ratio(input='{}/dir/'.format(dir),
                       output='{}/training_and_testing/'.format(dir),
                       seed=42, ratio=(.9, .1), group_prefix=None)
    print('train data & validation data saved in "data/training_and_testing" Folder..')


def make_train_data(dir):
    train_images = "{}/training_and_testing/train/images".format(dir)
    target_train_images = "{}/data/training_data/train_images/train".format(dir)
    train_masks = "{}/training_and_testing/train/masks".format(dir)
    target_train_masks = "{}/data/training_data/train_masks/train".format(dir)
    for img, msk in zip(os.listdir(train_images), os.listdir(train_masks)):
        num_img = img.split('.')[0]
        num_msk = msk.split('.')[0]
        try:
            if num_msk == num_img:
                image = cv2.imread(os.path.join(train_images, num_img+'.jpg'), 1)
                cv2.imwrite(os.path.join(target_train_images, num_img+'.jpg'), image)
                mask = cv2.imread(os.path.join(train_masks, num_img+'.png'), 0)
                cv2.imwrite(os.path.join(target_train_masks, num_img+'.png'), mask)
        except: pass
    print('training data generated...')


def make_validation_data(dir):
    val_images = "{}/training_and_testing/val/images".format(dir)
    target_val_images = "{}/data/training_data/val_images/val".format(dir)
    val_masks = "{}/training_and_testing/val/masks".format(dir)
    target_val_masks = "{}/data/training_data/val_masks/val".format(dir)
    for img, msk in zip(os.listdir(val_images), os.listdir(val_masks)):
        num_img = img.split('.')[0]
        num_msk = msk.split('.')[0]
        try:
            if num_msk == num_img:
                image = cv2.imread(os.path.join(val_images, num_img+'.jpg'), 1)
                cv2.imwrite(os.path.join(target_val_images, num_img+'.jpg'), image)
                mask = cv2.imread(os.path.join(val_masks, num_img+'.png'), 0)
                cv2.imwrite(os.path.join(target_val_masks, num_img+'.png'), mask)
        except: pass
    print('validation data generated...')


def showData(dir):
    path_mask = '{}/masks'.format(dir)
    path_img = '{}/images'.format(dir)
    mskLst = os.listdir(path_mask)
    imgLst = os.listdir(path_img)
    for msk, img in zip(mskLst, imgLst):
        mk = cv2.imread(os.path.join(path_mask, msk))
        ht, wd = mk.shape[:2]
        mk = cv2.resize(mk, (640, 640*ht//wd), interpolation=cv2.INTER_NEAREST)
        im = cv2.imread(os.path.join(path_img, img))
        im = cv2.resize(im, (640, 640*ht//wd), interpolation=cv2.INTER_NEAREST)
        join = np.concatenate((im, mk), axis=1)
        cv2.imshow('image-mask', join)
        cv2.waitKey(200)
    cv2.destroyAllWindows()


"""
code written by @swati sinha, @maitry sinha, @bibekananda sinha 
"""
