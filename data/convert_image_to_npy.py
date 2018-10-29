# Generate npy for train/val cityscape dataset

import numpy as np
import data_utils
from scipy.misc import imread
from glob import glob

if __name__ == "__main__":
    annotation_train_path = './gtFine/train'
    annotation_val_path = './gtFine/val'

    boundary_train_path = './edge/train'
    boundary_val_path = './edge/val'

    image_train_path = './leftImg8bit/train'
    image_val_path = './leftImg8bit/val'

    save_target_train_path = './npy/train/'
    save_target_val_path = './npy/val/'

    annotation_train_image_list = data_utils.find_imgs(annotation_train_path,encoding = "color")
    annotation_val_image_list = data_utils.find_imgs(annotation_val_path,encoding = "color")

    boundary_train_image_list = (glob(boundary_train_path+'/*.png'))
    boundary_train_image_list.sort()
    boundary_val_image_list = (glob(boundary_val_path+'/*.png'))
    boundary_val_image_list.sort()

    train_image_list = data_utils.find_imgs(image_train_path)
    val_image_list = data_utils.find_imgs(image_val_path)


    # For train images
    for (train_image_path, train_annotation_path, train_boundary_path) in zip(train_image_list, annotation_train_image_list, boundary_train_image_list):
        name = train_annotation_path.split('/')[-1]
        name = "_".join(name.split('_')[0:3])
        edge_name = train_boundary_path.split('/')[-1]
        edge_name = "_".join(edge_name.split('_')[:3])
	# check name is the same or not
        if name == edge_name:
            train_annotation = imread(train_annotation_path)[:,:,:3]
            train_image = imread(train_image_path)
            train_boundary = imread(train_boundary_path)
            train_boundary = train_boundary.reshape((train_boundary.shape)+(1,))
            print('Processing {} ....'.format(name))
            npdata = np.concatenate((train_image, train_annotation, train_boundary), axis = 2)
            np.save(save_target_train_path + name, npdata)
        else:
            print('NAME IS DIFFERENT!')

    # For validation images
    for (val_image_path, val_annotation_path, val_boundary_path) in zip(val_image_list, annotation_val_image_list, boundary_val_image_list):
        name = val_annotation_path.split('/')[-1]
        name = "_".join(name.split('_')[0:3])
        edge_name = val_boundary_path.split('/')[-1]
        edge_name = "_".join(edge_name.split('_')[:3])
	# check name is the same or not
        if name == edge_name:
            val_annotation = imread(val_annotation_path)[:,:,:3]
            val_image = imread(val_image_path)
            val_boundary = imread(val_boundary_path)
            val_boundary = train_boundary.reshape((val_boundary_path.shape)+(1,))
            print('Processing {} ....'.format(name))
            npdata = np.concatenate((val_image, val_annotation, val_boundary), axis = 2)
            np.save(save_target_val_path + name, npdata)
        else:
            print('NAME IS DIFFERENT!')
