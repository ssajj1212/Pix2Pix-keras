# data_utils
# include all function to process data
import os
from glob import glob
from scipy.misc import imread, imsave
import numpy as np





def extract_edge(img):
    edge_img = np.ones(img.shape)
    for ii in range(img.shape[0])[1:-1]:
        for jj in range(img.shape[1])[1:-1]:
            edge_img[ii,jj] = ((img[ii,jj] == img[ii-1,jj-1]) and (img[ii,jj] == img[ii-1,jj]) and (img[ii,jj] == img[ii-1,jj+1]) and (img[ii,jj] == img[ii,jj-1]) and (img[ii,jj] == img[ii,jj]) and (img[ii,jj] == img[ii,jj+1]) and (img[ii,jj] == img[ii-1,jj]) and (img[ii,jj] == img[ii-1,jj+1]) and (img[ii,jj] == img[ii+1,jj-1]) and (img[ii,jj] == img[ii+1,jj]) and (img[ii,jj] == img[ii+1,jj+1]))
    return edge_img


def find_imgs(path, encoding = "nonlabel"):
    dir_list = os.listdir(path)
    imgs_list = []
    for dir_ in dir_list:
        if encoding == "color":
            imgs_list.append(glob(path+'/'+dir_+'/*_color.png'))
        elif encoding == "instance":
            imgs_list.append(glob(path+'/'+dir_+'/*_instanceIds.png'))
        elif encoding == "labelid":
            imgs_list.append(glob(path+'/'+dir_+'/*_labelIds.png'))
        else:
            imgs_list.append(glob(path+'/'+dir_+'/*.png'))

    imgs_list = [item for sublist in imgs_list for item in sublist]

    imgs_list.sort()

    print('Found {} images'.format(len(imgs_list)))
    return imgs_list

def crop_img(img, target_dim = (256,256,3)):
    original_dim = img.shape
    num_columns = math.ceil(img.shape[1]/target_dim[1])
    num_rows = math.ceil(img.shape[0]/target_dim[0])   # i channels_first
    if not original_dim[0]%target_dim[0]:
        list_row_idx = [(i * target_dim[0], (i + 1) * target_dim[0]) for i in range(original_dim[0]//target_dim[0])]
    else:
        sliding_pixel = target_dim[0] - (target_dim[0] * num_rows - original_dim[0])//(num_rows-1)
        print('row sliding pixel {}'.format(sliding_pixel))
        list_row_idx = [(i * sliding_pixel, i * sliding_pixel + target_dim[0]) for i in range(num_rows)]
    if not original_dim[1]%target_dim[1]:
        list_col_idx = [(i * target_dim[1], (i + 1) * target_dim[1]) for i in range(original_dim[1] // target_dim[1])]
    else:
        sliding_pixel = target_dim[1] - (target_dim[1] * num_columns - original_dim[1])//(num_columns-1)
        print('column sliding pixel {}'.format(sliding_pixel))
        list_col_idx = [(i * sliding_pixel, i * sliding_pixel + target_dim[1]) for i in range(num_columns)]
    new_image_list = []
    for (index_row,(row_start_index,row_end_index)) in enumerate(list_row_idx):
        for (index_column, (column_start_index,column_end_index)) in enumerate(list_col_idx):
            new_image_list.append(img[row_start_index:row_end_index, column_start_index:column_end_index,:])
            index = index_row*num_rows + index_column
    return new_image_list
