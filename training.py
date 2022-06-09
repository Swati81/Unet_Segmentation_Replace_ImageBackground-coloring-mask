"""
code written by @swati sinha, @maitry sinha, @bibekananda sinha
"""
import tensorflow as tf
import os
import h5py
from unet import unet_model
from generator import trainGenerator, plot_history
from data import mask_labels_after_process
from preprocess import data_dir, img_size, batch


model_path = 'unet_models/unet_model_car_256.hdf5'


root_dir = data_dir()
batch_size = batch()
n_class = mask_labels_after_process(dir=root_dir)
size = img_size()


class training:
    def __init__(self, n_class=n_class, model_path=model_path, root_dir=root_dir, batch_size=batch_size):
        self.batch_size = batch_size
        self.n_class = n_class
        self.model_path = model_path
        self.directory = root_dir
        self.train_img_path = "{}/data/training_data/train_images/".format(self.directory)
        self.train_mask_path = "{}/data/training_data/train_masks/".format(self.directory)
        self.val_img_path = "{}/data/training_data/val_images/".format(self.directory)
        self.val_mask_path = "{}/data/training_data/val_masks/".format(self.directory)
        self.train_img_gen = trainGenerator(self.train_img_path, self.train_mask_path,
                                            batch_size=self.batch_size, n_class=self.n_class)
        self.val_img_gen = trainGenerator(self.val_img_path, self.val_mask_path,
                                          batch_size=self.batch_size, n_class=self.n_class)
        self.num_train_imgs = len(os.listdir(os.path.join(self.train_img_path, 'train')))
        self.num_val_imgs = len(os.listdir('{}/data/training_data/val_images/val/'.format(self.directory)))
        self.steps_per_epoch = self.num_train_imgs // self.batch_size
        self.val_steps_per_epoch = self.num_val_imgs // self.batch_size

    def print_gen(self):
        print('No. of train images     :', self.num_train_imgs)
        print('No. of validation images:', self.num_val_imgs)
        print('training steps per epoch:', self.steps_per_epoch)
        print('validation steps per epoch:', self.val_steps_per_epoch)

    def model_save(self, size=(size, size, 3), n_class=n_class):
        mod = unet_model(size=size, n_class=n_class)
        os.makedirs('unet_models', exist_ok=True)
        mod.save(self.model_path)
        print('model saved at....:{}'.format(self.model_path))

    def train(self, epochs=15, lr=2e-4, training_type='saved_model'):
        if training_type == 'saved_model':
            model = tf.keras.models.load_model(self.model_path, compile=True)
            print('saved model loaded..')
        elif training_type == 'new_model':
            self.model_save(size=(size, size, 3), n_class=n_class)
            model = tf.keras.models.load_model(self.model_path, compile=True)
            print('new model loaded...')

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss="categorical_crossentropy", metrics="accuracy")
        hist = model.fit(self.train_img_gen, steps_per_epoch=self.steps_per_epoch, epochs=epochs,
                         verbose=1, validation_data=self.val_img_gen,
                         validation_steps=self.val_steps_per_epoch)
        model.save(self.model_path)
        print('model trained and saved..')
        return hist


train = training(model_path=model_path, root_dir=root_dir, n_class=n_class, batch_size=batch_size)
# train.print_gen()
hist = train.train(epochs=10, lr=2e-4, training_type='saved_model')  # for new data/model write 'new_model'
plot_history(hist)
