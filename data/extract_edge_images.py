# Extract edge images
'''
Extract edge of instance annotation images
'''
from scipy.misc import imread, imsave
import data_utils


if __name__ == "__main__":
    target_save_train_path = './edge/train/'
    target_save_val_path = './edge/val/'
    annotation_train_path = './gtFine/train'
    annotation_val_path = './gtFine/val'


    annotation_train_image_list = find_imgs(annotation_train_path,encoding = "instance")
    annotation_val_image_list = find_imgs(annotation_val_path,encoding = "instance")

    for annotation_train_path in annotation_train_image_list:
        name = annotation_train_path.split('/')[-1]
        name = "_".join(name.split('_')[0:-1])
        name = name + '_boundaryMap.png'
        print('Processing {} ....'.format(name))
        train_annotation = imread(annotation_train_path)
        edge_train = data_utils.extract_edge(train_annotation)
        imsave(target_save_train_path+name,edge_train)

    for annotation_val_path in annotation_val_image_list:
        name = annotation_val_path.split('/')[-1]
        name = "_".join(name.split('_')[0:-1])
        name = name + '_boundaryMap.png'
        print('Processing {} ....'.format(name))
        val_annotation = imread(annotation_val_path)
        edge_val = extract_edge(val_annotation)
        imsave(target_save_val_path+name,edge_val)
