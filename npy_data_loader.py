
# Load npy data

# This version specifically for pix2pix, so load imgs_B only [H,W,3]

from glob import glob
import numpy as np


class NPY_DataLoader():
    def __init__(self, path, img_res = (1024,2048), data_type = "train"):
        self.img_res = img_res
        self._data_type = data_type
        if self._data_type == "train":
            self._is_training = True
            self._is_val = False
            self._is_testing = False
        elif self._data_type == "val" or self._data_type == "validation":
            self._is_training = False
            self._is_val = True
            self._is_testing = False
        elif self._data_type == "test":
            self._is_training = False
            self._is_val = False
            self._is_testing = True
        else:
            print("Typo, please specify data_type as train, val or test")
        self._path = path + '/%s' % (self._data_type)
        print('path: {}'.format(self._path))

        # Check whether there are images under the path
        if glob(self._path + '/*') == []:
            print("Please check dataset directory")
            return
        else:
            self._data_list = glob(self._path + '/*')
            self._data_list = self._data_list
            self._data_size = len(self._data_list)
            print('{} {} samples are found'.format(self._data_size,self._data_type))


        self.downsample_rate = int(np.load(self._data_list[0]).shape[0]/img_res[0])
        print('downsample rate {}'.format(self.downsample_rate))

    def get_data_size(self):
        return self._data_size

    def load_data(self, batch_size=0):
        '''
        Return small batch of whole dataset if batch_size is specified
        otherwise return whole dataset
        '''
        if batch_size == 0:
            batch_size = self._data_size

        batch_data_list = np.random.choice(self._data_list, size=batch_size)

        if not self.downsample_rate == 1:
            arrs = [np.load(npy_path)[::self.downsample_rate,::self.downsample_rate,] for npy_path in batch_data_list]
        else:
            arrs = [np.load(npy_path) for npy_path in batch_data_list]

        res = np.concatenate([arr[np.newaxis] for arr in arrs])

        # Imgs_A - real image, imgs_B - color annotation + boundary map
        imgs_A = res[:,:,:,:3]/127.5 - 1. # Normalize to [-1,1]
        imgs_B = res[:,:,:,3:6]/127.5 - 1. # Normalize to [-1,1]
        #imgs_B = np.concatenate((imgs_B,res[:,:,:,6].reshape(res.shape[:3]+(1,))),axis=3)

        return imgs_A, imgs_B

    def load_batch(self, batch_size=0):
        '''
        Load batches of data, return a generator to save memory
        '''
        if batch_size == 0:
            batch_size = self._data_size

        self.n_batches = int(len(self._data_list) / batch_size)

        for i in range(self.n_batches-1):
            batch_data_list = self._data_list[i*batch_size:(i+1)*batch_size]

            if not self.downsample_rate == 1:
                arrs = [np.load(npy_path)[::self.downsample_rate,::self.downsample_rate,] for npy_path in batch_data_list]
            else:
                arrs = [np.load(npy_path) for npy_path in batch_data_list]

            res = np.concatenate([arr[np.newaxis] for arr in arrs])

            imgs_A = res[:,:,:,:3]/127.5 - 1. # Normalize to [-1,1]
            imgs_B = res[:,:,:,3:6]/127.5 - 1. # Normalize to [-1,1]
            #imgs_B = np.concatenate((imgs_B,res[:,:,:,6].reshape(res.shape[:3]+(1,))),axis=3)
]

            yield imgs_A, imgs_B



# if __name__ == "__main__":
#     npy_file_path = '/home/cshuangs/Cityscapes/npy'
#     train_data_loader = NPY_DataLoader(npy_file_path)
#     train_data_loader.load_data()
