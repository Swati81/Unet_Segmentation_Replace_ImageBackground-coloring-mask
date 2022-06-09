"""
code written by @swati sinha, @maitry sinha, @bibekananda sinha
"""
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from data import mask_labels_after_process
from preprocess import img_size, data_dir, batch

root_dir = data_dir()
size = img_size()
batch_size = batch()
n_class = mask_labels_after_process(dir=root_dir)


def preprocess_data(img, mask, n_class=n_class):
    # for pet mask
    img = (img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = img / 255.0
    # make one-hot encode
    mask = to_categorical(mask, num_classes=n_class)
    return img, mask


def trainGenerator(train_img_path, train_mask_path, batch_size=batch_size, img_size=size, n_class=n_class):
    img_data_gen_args = dict(horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='reflect')

    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode=None,
        target_size=(img_size, img_size),
        color_mode='rgb',
        batch_size=batch_size,
        seed=200)

    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode=None,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        batch_size=batch_size,
        seed=200)

    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, n_class=n_class)
        yield img, mask


def draw(x, y):
    image = x[0]
    mask = y[0]
    plt.figure(figsize=(16, 10))
    plt.subplot(121)
    plt.imshow(image)
    plt.title('Image')
    plt.subplot(122)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.imshow(mask)
    plt.show()


def plot_history(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.show()


# ### to check generator uncomment bellow #################################
"""
train_img_path = "{}/data/training_data/train_images/".format(root_dir)
train_mask_path = "{}/data/training_data/train_masks/".format(root_dir)
val_img_path = "{}/data/training_data/val_images/".format(root_dir)
val_mask_path = "{}/data/training_data/val_masks/".format(root_dir)

train_img_gen = trainGenerator(train_img_path, train_mask_path)
val_img_gen = trainGenerator(val_img_path, val_mask_path)
val_x, val_y = val_img_gen.__next__()
x, y = train_img_gen.__next__()
print(x.shape)
print(y.shape)
print(np.unique(y))
"""
