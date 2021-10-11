
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np
import random
from matplotlib import pyplot as plt


class CFRDatagenerator(tf.keras.utils.Sequence):
    """
    Custom data generator to generate batches of siamsese pairs for training
    and validation'
    """
    def __init__(self, dataset_dir, batch_size=4, no_of_batches_per_epoch=10,
                 input_dims=(60, 60, 3)):
        self.dataset_dir = dataset_dir
        self.input_dims = input_dims
        self.input_rows = self.input_dims[0]
        self.input_cols = self.input_dims[1]
        self.no_of_channels = self.input_dims[2]
        self.batch_size = batch_size
        self.no_of_batches_per_epoch = no_of_batches_per_epoch
        self.class_ids = os.listdir(self.dataset_dir)
        self.total_ids = len(self.class_ids)
        self.images_to_ids = {}
        self.prepare_data()
        self.on_epoch_end()

    def prepare_data(self):
        for class_id in self.class_ids:
            if class_id not in self.images_to_ids.keys():
                self.images_to_ids[class_id] = []
            images_for_this_id = os.listdir(
                os.path.join(self.dataset_dir, class_id, 'frontal'))
            self.images_to_ids[class_id] = [
                sample for sample in images_for_this_id]

    def __len__(self):
        """ No of batches per epoch."""
        return self.no_of_batches_per_epoch

    def on_epoch_end(self):
        pass

    def __data_generation__(self, _):
        X = [np.empty((self.batch_size, self.input_rows, self.input_cols,
                       self.no_of_channels)) for i in range(2)]
        y = np.empty(self.batch_size)
        y[:self.batch_size//2] = 1
        y[self.batch_size//2:] = 0

        reference_id = random.choice(self.class_ids)
        negative_ids = [class_id for class_id in
                        self.class_ids if class_id != reference_id]

        target_size = (self.input_rows, self.input_cols)
        # prepariong positive pairs, half the batch_size pairs will be positive
        for i in range(self.batch_size//2):
            current_image_name = random.choice(
                self.images_to_ids[reference_id])
            current_image_path = os.path.join(
                self.dataset_dir, reference_id, 'frontal', current_image_name)
            
            positive_image_name = random.choice(
                self.images_to_ids[reference_id])
            positive_image_path = os.path.join(
                self.dataset_dir, reference_id, 'frontal', positive_image_name)
            
            current_image = img_to_array(load_img(
                current_image_path, target_size=target_size))
            positive_image = img_to_array(load_img(
                positive_image_path, target_size=target_size))

            # X.append([current_image, positive_image])
            # y.append([1])
            X[0][i] = current_image
            X[1][i] = positive_image

            negative_id = random.choice(negative_ids)
            negative_image_name = random.choice(
                self.images_to_ids[negative_id])
            negative_image_path = os.path.join(
                self.dataset_dir, negative_id, 'frontal', negative_image_name)
            negative_image = img_to_array(load_img(
                negative_image_path, target_size=target_size))
            
            X[0][i+self.batch_size//2] = current_image
            X[1][i+self.batch_size//2] = negative_image

        # return np.array(X), np.array(y)
        # X = X / 255.
        return X, y

    def __getitem__(self, index):
        X, y = self.__data_generation__(self)
        return X, y


def plot_pairs(pairs_of_images, labels):
    number_of_rows = len(labels)
    fig, axs = plt.subplots(nrows=number_of_rows, ncols=2)
    for i in range(number_of_rows):
        # axs[i, 0].set_axis_off()
        # axs[i, 1].set_axis_off()
        axs[i, 0].get_xaxis().set_ticklabels([])
        axs[i, 1].get_xaxis().set_ticklabels([])
        axs[i, 0].get_yaxis().set_ticklabels([])
        axs[i, 1].get_yaxis().set_ticklabels([])

        axs[i, 0].set_ylabel(labels[i], rotation=0)
        axs[i, 0].matshow(np.asarray(pairs_of_images[0][i]).astype('uint8'))
        axs[i, 1].matshow(np.asarray(pairs_of_images[1][i]).astype('uint8'))
    plt.tight_layout()
    plt.savefig('training_pairs.png', bbox_inches='tight')
    plt.show()
