"""
code written by @swati sinha, @maitry sinha, @bibekananda sinha
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from preprocess import img_size

size = img_size()


class predict:
    def __init__(self, model_path, img):
        self.model_path = model_path
        self.img = img[:, :, ::-1]  # rgb

    def loading_model(self):
        model = tf.keras.models.load_model(self.model_path, compile=False)
        return model

    def prediction(self):
        model = self.loading_model()
        ht, wd = self.img.shape[:2]
        image = cv2.resize(self.img, (size, size), interpolation=cv2.INTER_NEAREST)
        image = np.expand_dims(image, axis=0).copy()
        image = image / 255.0
        predicted_mask = model.predict(image)
        predicted_mask = tf.argmax(predicted_mask, axis=-1)
        predicted_mask = predicted_mask[..., tf.newaxis]
        mask = np.array(predicted_mask[0], dtype='uint8')
        mask = cv2.resize(mask, (wd, ht), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=-1)
        # making 3D mask from 2D
        mask = np.dstack((mask, mask, mask))
        return mask


def color_mask(mask):
    colors = np.random.uniform(low=0, high=100, size=(59, 3))
    # changing BG-black to BG-white
    mask[:, :, 0][mask[:, :, 0] == 0] = 206
    mask[:, :, 1][mask[:, :, 1] == 0] = 232
    mask[:, :, 2][mask[:, :, 2] == 0] = 190
    for i, col in enumerate(colors):
        mask[:, :, 0][mask[:, :, 0] == i + 1] = int(col[0])
        mask[:, :, 0][mask[:, :, 0] == i + 1] = int(col[2])
        mask[:, :, 0][mask[:, :, 0] == i + 1] = int(col[1])
    # join = np.concatenate((self.img, col_mask), axis=1)
    return mask[:, :, ::-1]


def replace_image(mask, img, type='RI'):
    if type == 'RI':
        output = np.where(mask, mask, img)
    elif type == 'RB':
        output = np.where(mask, img, mask)
        output[output == 0] = 255
    # join = np.concatenate((self.img, output), axis=1)
    elif type == 'WM':
        output = np.where(mask, img, mask*255)
    return output


# ######## show in video #################
"""
vid_path = "fashion.mp4"
def video(self, vid_path):
    cap = cv2.VideoCapture(vid_path)
    while True:
        ret, img = cap.read()
        image = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
        frame = image[0:360, 140:400].copy()
        try:
            p = predict(model_path=model_path, img=frame)
            p.replace_image()
            cv2.imshow('FASHION', fram)
            if cv2.waitKey(2) == 13:
                break
                cap.release()
                cv2.destroyAllWindows()
        except:
            cv2.imshow('FASHION', frame)
            if cv2.waitKey(2) == 13:
                break
                cap.release()
                cv2.destroyAllWindows()

"""
