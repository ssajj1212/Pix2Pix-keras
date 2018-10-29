# Modified data_utils.py
# originated from https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/pix2pix/src/utils/data_utils.py
# Copyright belongs to original author



from keras.utils import np_utils
import numpy as np
import h5py
from scipy.misc import imsave

import matplotlib.pylab as plt


def normalization(X):
    return X / 127.5 - 1


def inverse_normalization(X):

    return (X + 1.) * 127.5


def get_nb_patch(img_dim, patch_size, image_data_format):

    assert image_data_format in ["channels_first", "channels_last"], "Bad image_data_format"

    if image_data_format == "channels_first":
        assert img_dim[1] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[2] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] // patch_size[1])
        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])

    elif image_data_format == "channels_last":
        assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
        img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])

    return nb_patch, img_dim_disc


def extract_patches(X, image_data_format, patch_size, overlapping_rate):

    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        X = X.transpose(0,2,3,1)

    patch_height = patch_size[0]
    patch_width = patch_size[1]

    list_X = []

    img_rows = X.shape[1]
    img_cols = X.shape[2]

    num_col = int(img_cols/overlapping_rate /patch_width) - 1
    num_row = int(img_rows/overlapping_rate /patch_height) - 1

    list_row_idx = [( int(i * overlapping_rate * patch_height), int((i * overlapping_rate + 1)* patch_height)) for i in range(num_row)]
    list_col_idx = [( int(i * overlapping_rate * patch_width), int((i * overlapping_rate + 1) * patch_width)) for i in range(num_col)]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)

    return list_X



def get_disc_batch(X_batch, img_type, patch_size, image_data_format, overlapping_rate):

    # real imgs
    if img_type == "real":
        y_disc_real = np.zeros((X_batch.shape[0],2), dtype = np.uint8)
        y_disc_real[:, 1] = 1
        X_disc_real = extract_patches(X_batch, image_data_format, patch_size, overlapping_rate)
        X_disc = X_disc_real
        y_disc = y_disc_real

    # fake imgs

    elif img_type == "fake":
        y_disc_fake = np.zeros((X_batch.shape[0],2), dtype = np.uint8)
        y_disc_fake[:, 0] = 1
        X_disc_fake = extract_patches(X_batch, image_data_format, patch_size, overlapping_rate)
        X_disc = X_disc_fake
        y_disc = y_disc_fake

    return X_disc, y_disc


def plot_generated_batch(X_full, X_sketch, X_gen, batch_size, image_data_format, data_type, suffix, path):

    X_sketch = inverse_normalization(X_sketch).astype(np.uint8)
    X_full = inverse_normalization(X_full).astype(np.uint8)
    X_gen = inverse_normalization(X_gen).astype(np.uint8)

    Xs = X_sketch[:96]
    Xg = X_gen[:96]
    Xr = X_full[:96]

    if image_data_format == "channels_last":

        list_rows = []
        for i in range(int(Xs.shape[0] // 16)):
            X = np.concatenate([Xs[k] for k in range(16 * i, 16 * (i + 1))], axis=1)
            list_rows.append(X)
            X = np.concatenate([Xg[k] for k in range(16 * i, 16 * (i + 1))], axis=1)
            list_rows.append(X)
            X = np.concatenate([Xr[k] for k in range(16 * i, 16 * (i + 1))], axis=1)
            list_rows.append(X)

        X = np.concatenate(list_rows, axis=0)

    if image_data_format == "channels_first":

        list_rows = []
        for i in range(int(Xs.shape[0] // 16)):
            X = np.concatenate([Xs[k] for k in range(16 * i, 16 * (i + 1))], axis=2)
            list_rows.append(X)
            X = np.concatenate([Xg[k] for k in range(16 * i, 16 * (i + 1))], axis=2)
            list_rows.append(X)
            X = np.concatenate([Xr[k] for k in range(16 * i, 16 * (i + 1))], axis=2)
            list_rows.append(X)

        X = np.concatenate(list_rows, axis=1)
        X = Xr.transpose(1,2,0)

    # Use imsave to flush out the result instead of using pyplot (improve resolution for output)
    imsave(path+"/epoch_" + data_type + "%s.png" % suffix, X)
