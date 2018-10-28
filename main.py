import os
from pix2pix import Pix2Pix

if __name__ == "__main__":
    DATA_PATH = './data/npy'
    NUM_EPOCH = 1000

    pix2pix = Pix2Pix(img_dim = (128, 256, 3), batch_size = 16, patch_size = 64)
    print('Training.....')
    pix2pix.train(num_epoch = NUM_EPOCH, data_path = DATA_PATH)
