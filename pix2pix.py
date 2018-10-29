from npy_data_loader import NPY_DataLoader
from keras.optimizers import Adam, SGD, Adagrad
from keras.utils import generic_utils
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Input, Concatenate, average
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.utils import plot_model
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.callbacks import TensorBoard
import numpy as np
import datetime
import time
import data_utils
import os
import logging
import pandas as pd

class Pix2Pix():
    def __init__(self, img_dim = (256,256,3),batch_size = 16,
                  data_format = "channels_last", patch_size = 64):
        # initialize Pix2Pix instance
        assert K.backend() == "tensorflow", "Not implemented with theano backend"
        self._data_format = data_format
        # check data dimension
        if self._data_format == "channels_first":
            # channel first [channel, height(row), width(column)]
            self.img_rows, self.img_cols = img_dim[1:]
            self.img_channels = img_dim[0]
            self.img_dim = (self.img_channels, self.img_rows, self.img_cols)
        elif self._data_format == "channels_last":
            # channels_last [height, width, channel] default
            self.img_rows, self.img_cols = img_dim[:-1]
            self.img_channels = img_dim[2]
            self.img_dim = (self.img_rows, self.img_cols, self.img_channels)
        else:
            print('data format is specified wrongly, please use channels_first or channels_last')



        self.batch_size = batch_size
        if self._data_format == "channels_first":
            self.patch_dim = (3,patch_size, patch_size)
        else:
            self.patch_dim = (patch_size, patch_size, 3)

        # Add overlapping patch feature
        OVERLAPPING_RATE = 0.5
        self.patch_overlapping_rate = OVERLAPPING_RATE
        num_col = int(self.img_cols/OVERLAPPING_RATE/patch_size) - 1
        num_row = int(self.img_rows/OVERLAPPING_RATE/patch_size) - 1

        self.num_patch = int(num_col * num_row)

        self.Initialize_dir_files()

        # Define and compile generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()



        self.CGAN_model = self.build_CGAN()

        # Define optimizer for discriminator and cGAN`
        self.optimizer_cgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.optimizer_discriminator = SGD(lr=1E-3, momentum=0.9, decay=1e-6, nesterov=False)



    def Initialize_dir_files(self):
        Training_file_name = str(datetime.datetime.now())
        Training_file_name = str(datetime.datetime.now())
        os.chdir('logs')
        os.mkdir(Training_file_name)
        os.chdir('../figures')
        os.mkdir(Training_file_name)
        os.chdir('../saved_model')
        os.mkdir(Training_file_name)
        os.chdir('..')
        self.figure_path = './figures/'+ Training_file_name
        self.model_save_path = './saved_model/'+ Training_file_name
        self.log_path = './logs/' + Training_file_name
        open(self.log_path+'/training_log.log', 'a').close() # Create file
        logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=self.log_path+'/training_log.log',
                    filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)

        self.logger_model = logging.getLogger('Model info')
        self.logger_traininfo = logging.getLogger('Training info')



    def build_CGAN(self):
        '''
        Combine discriminator and generator to create DCGAN model
        '''

        # TO DO: Check generator, discriminator is initiated properly

        # Compile generator & discriminator
        self.discriminator.trainable = False #unable trainable for discriminator before merge generator, discriminator together

        # Initiate DCGAN to merge generator and discriminator
        generator_input = Input(shape=self.img_dim, name = "CGAN_input")
        generated_image = self.generator(generator_input)
        print('patch dim : {}'.format(self.patch_dim))
        if self._data_format == "channels_first":
            patch_height, patch_width = self.patch_dim[1:]
            img_height, img_width = self.img_dim[1:]
        else:
            patch_height, patch_width = self.patch_dim[:-1]
            img_height, img_width = self.img_dim[:-1]

        print('patch height: {}, patch width: {}'.format(patch_height,patch_width))
        print('image height: {}, image width: {}'.format(img_height, img_width))

        # Develop overlapping patch features to mitigate the patch boundary effect on generated image
        num_col = int(self.img_cols/self.patch_overlapping_rate /patch_width) - 1
        num_row = int(self.img_rows/self.patch_overlapping_rate /patch_height) - 1


        list_row_idx = [( int(i * self.patch_overlapping_rate * patch_height), int((i * self.patch_overlapping_rate + 1)* patch_height)) for i in range(num_row)]
        list_col_idx = [( int(i * self.patch_overlapping_rate * patch_width), int((i * self.patch_overlapping_rate + 1) * patch_width)) for i in range(num_col)]

        # Get all generated patches tensor
        list_gen_patch = []
        for row_idx in list_row_idx:
            for col_idx in list_col_idx:
                if self._data_format == "channels_last":
                    x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(generated_image)
                else:
                    x_patch = Lambda(lambda z: z[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]])(generated_image)
                list_gen_patch.append(x_patch)

        discriminator_output = self.discriminator(list_gen_patch)

        CGAN_model = Model(inputs = [generator_input],
                            outputs = [generated_image, generated_image, discriminator_output],
                            name = 'CGAN')
        print('CGAN model summary:')
        print(CGAN_model.summary())


        return CGAN_model



    def train(self, num_epoch, data_path):# dataset_name, data_path = './datasets'):
        '''
        Train CGAN
        :param num_epoch: number of epoches specified by user

        '''
        def perceptual_loss(img_true, img_generated):
            image_shape = self.img_dim
            vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
            loss_block3 = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
            loss_block3.trainable = False
            loss_block2 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv2').output)
            loss_block2.trainable = False
            loss_block1 = Model(input=vgg.input, outputs = vgg.get_layer('block1_conv2').output)
            loss_block1.trainable = False
            return K.mean(K.square(loss_block1(img_true) - loss_block1(img_generated))) + 2*K.mean(K.square(loss_block2(img_true) - loss_block2(img_generated))) + 5*K.mean(K.square(loss_block3(img_true) - loss_block3(img_generated)))

        # Load data
        self.load_data(path = data_path)

        if self._data_format == "channels_first":
            patch_size = self.patch_dim[1:]
        else:
            patch_size = self.patch_dim[:2]

        self.discriminator.trainable = False
        self.generator.compile(loss=perceptual_loss , optimizer= self.optimizer_cgan)#,
                                #callbacks = [generator_tensorboard])


        # -------------
        # Compile CGAN
        # -------------
        CGAN_loss = ['mae', perceptual_loss, 'binary_crossentropy']
        CGAN_loss_weights = [2E-1, 2E-3, 1] # implement like equation 4 (lamda = 10)
        self.logger_model.info('Weights between L1 loss, perceptual loss and GAN loss is {}'.format(CGAN_loss_weights))


        self.CGAN_model.compile(loss = CGAN_loss, loss_weights = CGAN_loss_weights,
                            optimizer = self.optimizer_cgan)# ,callbacks = [CGAN_tensorboard])

        self.discriminator.trainable = True #enable trainable for discriminator
        self.discriminator.compile(loss="binary_crossentropy",
                                    optimizer = self.optimizer_discriminator)#,
                                    #callbacks = [discriminator_tensorboard])


        # Prepare to write log file

        self.logger_traininfo.info('Image size:                                           {0} '.format(self.img_dim))
        self.logger_traininfo.info('Patch size:                                           {0} '.format(self.patch_dim))
        self.logger_traininfo.info('Number of patches for image:                          {0} '.format(self.num_patch))
        self.logger_traininfo.info('Data format:                                          {0} '.format(self._data_format))

        self.logger_model.info('Generator summary:')
        self.generator.summary(print_fn=self.logger_model.info)
        self.logger_model.info('Discriminator summary:')
        self.discriminator.summary(print_fn=self.logger_model.info)
        self.logger_model.info('CGAN summary:')
        self.CGAN_model.summary(print_fn=self.logger_model.info)

        # Save model structure
        # Write json and hdf5 to save model
        model_json = self.discriminator.to_json()
        with open(self.model_save_path+'/'+"discriminator.json", "w") as json_file:
            json_file.write(model_json)
        model_json = self.generator.to_json()
        with open(self.model_save_path+'/'+"generator.json", "w") as json_file:
            json_file.write(model_json)
        model_json = self.CGAN_model.to_json()
        with open(self.model_save_path+'/'+"CGAN.json", "w") as json_file:
            json_file.write(model_json)

        self.logger_traininfo.info('Training start at {}.......'.format(datetime.datetime.now()))

        training_history = {'Epoch':[],'Batch':[],'D_logloss_train_real':[],'D_logloss_train_fake':[],'G_tot_loss_train':[], 'G_l1loss_train':[], 'G_ploss_train':[], 'G_ganloss_train':[],
                                                  'D_logloss_val_real':[], 'D_logloss_val_fake':[], 'G_tot_loss_val':[], 'G_l1loss_val':[], 'G_ploss_val':[], 'G_ganloss_val':[]}



        for epoch_iterator in range(num_epoch):
            epoch_training_start_time = time.time()
            # For each batch iteration, one epoch iterate all the data once unfortunately
            # data_loader load each batch in (imgs_A, imgs_B)
            for batch_i, (imgs_A, imgs_B) in enumerate(self.train_data_loader.load_batch(self.batch_size)):

                training_history['Epoch'].append(epoch_iterator)
                training_history['Batch'].append(batch_i)
                self.logger_traininfo.info('{}                 Epoch {}                  Batch {}'.format(datetime.datetime.now(),training_history['Epoch'][-1], training_history['Batch'][-1]))


                # -------------------
                # Train discriminator
                # -------------------

                imgs_A_fake = self.generator.predict(imgs_B)

                generated_patchs_real, generated_labels_real = data_utils.get_disc_batch(imgs_A, "real", patch_size, self._data_format, self.patch_overlapping_rate)
                generated_patchs_fake, generated_labels_fake = data_utils.get_disc_batch(imgs_A_fake, "fake", patch_size, self._data_format, self.patch_overlapping_rate)

                # Update discriminator
                if batch_i % 3 == 2:
                    self.discriminator.trainable = True
                    discriminator_loss_real = self.discriminator.train_on_batch(generated_patchs_real, generated_labels_real) # return binary_crossentropy loss for real img
                    discriminator_loss_fake = self.discriminator.train_on_batch(generated_patchs_fake, generated_labels_fake) # return binary_crossentropy loss for fake img
                    training_history['D_logloss_train_real'].append(discriminator_loss_real)
                    training_history['D_logloss_train_fake'].append(discriminator_loss_fake)

                    self.logger_traininfo.info('Discriminator logloss of real image on training                   {}'.format(training_history['D_logloss_train_real'][-1]))
                    self.logger_traininfo.info('Discriminator logloss of fake image on training                   {}'.format(training_history['D_logloss_train_fake'][-1]))

                else:
                    training_history['D_logloss_train_real'].append(None)
                    training_history['D_logloss_train_fake'].append(None)

                self.CGAN_model.trainable = True
                self.discriminator.trainable = False

                # ---------------
                # Train generator
                # ---------------

                # Train generator only with real image
                labels = np.zeros((imgs_A.shape[0],2),dtype=np.uint8) # one-hot encoding
                labels[:,1] = 1
                genarator_loss = self.CGAN_model.train_on_batch(imgs_B, [imgs_A, imgs_A, labels])

                training_history['G_tot_loss_train'].append(genarator_loss[0])
                training_history['G_l1loss_train'].append(genarator_loss[1])
                training_history['G_ploss_train'].append(genarator_loss[2])
                training_history['G_ganloss_train'].append(genarator_loss[3])


                self.logger_traininfo.info('Generator total loss on training                    {}'.format(training_history['G_tot_loss_train'][-1]))
                self.logger_traininfo.info('Generator L1 loss on training                       {}'.format(training_history['G_l1loss_train'][-1]))
                self.logger_traininfo.info('Generator peceptual loss on training                {}'.format(training_history['G_ploss_train'][-1]))
                self.logger_traininfo.info('Generator log GAN loss on training                  {}'.format(training_history['G_ganloss_train'][-1]))

                # After training 30 batch, test performance on validation data
                if batch_i % 30 == 29:

                    imgs_A_real_val, imgs_B_val = self.val_data_loader.load_data() # load whole data

                    # Get all patches
                    imgs_A_fake_val = self.generator.predict(imgs_B_val)

                    generated_patchs_real_val, generated_labels_real_val = data_utils.get_disc_batch(imgs_A_real_val, "real", patch_size, self._data_format, self.patch_overlapping_rate)
                    generated_patchs_fake_val, generated_labels_fake_val = data_utils.get_disc_batch(imgs_A_fake_val, "fake", patch_size, self._data_format, self.patch_overlapping_rate)

                    discriminator_loss_real_val = self.discriminator.evaluate(generated_patchs_real_val,generated_labels_real_val)
                    discriminator_loss_fake_val = self.discriminator.evaluate(generated_patchs_fake_val,generated_labels_fake_val)

                    training_history['D_logloss_val_real'].append(discriminator_loss_real_val)
                    training_history['D_logloss_val_fake'].append(discriminator_loss_fake_val)

                    self.logger_traininfo.info('Discriminator logloss of real image on validation                 {}'.format(training_history['D_logloss_val_real'][-1]))
                    self.logger_traininfo.info('Discriminator logloss of fake image on validation                 {}'.format(training_history['D_logloss_val_fake'][-1]))

                    labels_val = np.zeros((imgs_A_real_val.shape[0],2),dtype=np.uint8) # one-hot encoding
                    labels_val[:,1] = 1
                    generator_loss_validation = self.CGAN_model.evaluate(imgs_B_val, [imgs_A_real_val, imgs_A_real_val, labels_val])


                    training_history['G_tot_loss_val'].append(generator_loss_validation[0])
                    training_history['G_l1loss_val'].append(generator_loss_validation[0])
                    training_history['G_ploss_val'].append(generator_loss_validation[1])
                    training_history['G_ganloss_val'].append(generator_loss_validation[2])



                    self.logger_traininfo.info('Generator total loss on validation                  {}'.format(training_history['G_tot_loss_val'][-1]))
                    self.logger_traininfo.info('Generator L1 loss on validation                     {}'.format(training_history['G_l1loss_val'][-1]))
                    self.logger_traininfo.info('Generator peceptual loss on validation              {}'.format(training_history['G_ploss_val'][-1]))
                    self.logger_traininfo.info('Generator log GAN loss on validation                {}'.format(training_history['G_ganloss_val'][-1]))
                else:
                    training_history['D_logloss_val_real'].append(None)
                    training_history['D_logloss_val_fake'].append(None)
                    training_history['G_tot_loss_val'].append(None)
                    training_history['G_l1loss_val'].append(None)
                    training_history['G_ploss_val'].append(None)
                    training_history['G_ganloss_val'].append(None)

            time_elapse = time.time() - epoch_training_start_time # Unit [s]
            self.logger_traininfo.info('Epoch {0} costs {1}s'.format(epoch_iterator, time_elapse))

            #print('Epoch {0} completes! It costs {1}s\n'.format(epoch_iterator,time_elapse), file = file_obj)

            # Save generated samples for visualization every epoch
            imgs_A_real_train, imgs_B_train = self.train_data_loader.load_data()
            imgs_A_real_val, imgs_B_val = self.val_data_loader.load_data()

            imgs_A_fake_train = self.generator.predict(imgs_B_train)
            imgs_A_fake_val = self.generator.predict(imgs_B_val)

            data_utils.plot_generated_batch(imgs_A_real_train, imgs_B_train, imgs_A_fake_train, self.batch_size, self._data_format, 'training', epoch_iterator, self.figure_path)
            data_utils.plot_generated_batch(imgs_A_real_val, imgs_B_val, imgs_A_fake_val, self.batch_size, self._data_format, 'validation', epoch_iterator, self.figure_path)

            # Save training history in csv
            df = pd.DataFrame(training_history, columns = ['Epoch','Batch','D_logloss_train_real','D_logloss_train_fake','G_tot_loss_train', 'G_l1loss_train','G_ploss_train',
                                                         'G_ganloss_train','D_logloss_val_real', 'D_logloss_val_fake', 'G_tot_loss_val', 'G_l1loss_val',
                                                         'G_ploss_val','G_ganloss_val'])
            df.to_csv(self.log_path + '/training_history.csv')

            # Save weights of models
            if epoch_iterator % 10 == 9:
                self.generator.save_weights(self.model_save_path+'/'+'generator.hdf5', overwrite = True)
                self.discriminator.save_weights(self.model_save_path+'/'+'discriminator.hdf5', overwrite = True)
                self.CGAN_model.save_weights(self.model_save_path+'/'+'CGAN.hdf5', overwrite = True)



    def load_data(self, path):
        '''
        Load data
        '''
        try:
            self.train_data_loader = NPY_DataLoader(path, img_res =  (self.img_rows, self.img_cols))
        except:
            print("Training data fails to load")

        try:
            self.val_data_loader = NPY_DataLoader(path, img_res = (self.img_rows, self.img_cols), data_type = "val")
        except:
            print("Validation data fails to load")


    def build_generator(self,is_deconv = True):
        '''
        Create U-net skip connection generator
        encoder: C64-C128-C256-C512-C512-C512-C512-C512
                 Exceptions: BatchNorm not apply to first C64 layer

        U-net decoder: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
                       followed by Conv2D(output_dim) and Tanh (3 in general, 2 for colorization)

        Ck - Conv2D-BatchNorm-LeakyReLU with k (4,4) Conv2D filters,
        CDk - Conv2D-BatchNorm-Dropout-ReLU with k (4,4) Conv2D filters
        Encoder downsample by factor of 2
        Decoder upsample by factor of 2 (using UpSample2D or Deconv2D)
        LeakyReLU in encoder with leaky slope of 0.2

        '''

        # ----------------------------------------------------
        # Build encoder e.g C64-C128-C256-C512-C512-C512-C512-C512
        # ----------------------------------------------------

        switcher = {16: ["C64","C128","C256","C512"],
                    32: ["C64","C128","C256","C512","C512"],
                    64: ["C64","C128","C256","C512","C512","C512"],
                    128: ["C64","C128","C256","C512","C512","C512","C512"],
                    256: ["C64","C128","C256","C512","C512","C512","C512","C512"],
                    512: ["C64","C128","C256","C512","C512","C512","C512","C512","C512"]}

        unet_encoder_filters = switcher.get(min(self.img_rows, self.img_cols), "Invalid img size") # ensure the last activation tensor the shape
        self.logger_model.info('UNet generator encoder filters : {}'.format(unet_encoder_filters))

        unet_input_layer = Input(shape=self.img_dim, name="UNet_input")
        encoder = unet_input_layer
        encoder_list = [encoder]
        for encoder_i, encoder_filter in enumerate(unet_encoder_filters):
            filters = int(encoder_filter[1:])
            if encoder_i == 0:
                encoder = self.conv_block(encoder, filters, encoder_index = encoder_i, filter_name = encoder_filter, is_BatchNorm = False)
            else:
                encoder = self.conv_block(encoder, filters, encoder_index = encoder_i, filter_name = encoder_filter)
            encoder_list.append(encoder)

        # ------------------------------------------------------------
        # Build decoder CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        # ------------------------------------------------------------

        switcher = {16: ["CD512","C512","C256","C128"],
                    32: ["CD512","CD1024","C512","C256","C128"],
                    64: ["CD512","CD1024","C1024","C512","C256","C128"],
                    128: ["CD512","CD1024","C1024","C1024","C512","C256","C128"],
                    256: ["CD512","CD1024","CD1024","C1024","C1024","C512","C256","C128"],
                    512: ["CD512","CD1024","CD1024","C1024","C1024","C1024","C512","C256","C128"]}


        unet_decoder_filters = switcher.get(min(self.img_rows, self.img_cols), "Invalid img size") # ensure the last activation tensor the shape
        self.logger_model.info('UNet generator decoder filters: {}'.format(unet_decoder_filters))

        decoder =  encoder_list[-1]
        decoder_list = [decoder]
        for decoder_i, decoder_filter in enumerate(unet_decoder_filters):
            if decoder_filter[0] == 'C' and not decoder_filter[0:2] == "CD":
                # Ck - without Dropout`
                is_dropout = False
                filters = int(decoder_filter[1:])
            elif decoder_filter[0:2] == "CD":
                is_dropout = True
                filters = int(decoder_filter[2:])

            if decoder_i == len(unet_decoder_filters)-1 or decoder_i == 0:
                is_concatenate = False
            else:
                is_concatenate = True

            if is_deconv:
                # build decoder using deconv_block
                decoder = self.deconv_block(decoder, encoder_list[-decoder_i-2], filters, decoder_index = decoder_i,
                                         filter_name = decoder_filter, is_BatchNorm = True, dropout = is_dropout, concatenate = is_concatenate)
            else:
                decoder = self.up_conv_block(decoder, encoder_list[-decoder_i-2], filters, decoder_index = decoder_i,
                                         filter_name = decoder_filter, is_BatchNorm = True, dropout = is_dropout, concatenate = is_concatenate)
            decoder_list.append(decoder)
            print('decoder tensor {} is passed'.format(decoder))

        # Last two layers Conv(output_dim), Tanh
        decoder = Conv2D(3,(4,4), name = 'Decoder_' + str(decoder_i+1) + '_Conv_final',
                        padding = "same", data_format = self._data_format)(decoder)
        decoder_list.append(decoder)
        decoder = Activation("tanh", name = 'Decoder_' + str(decoder_i+2) + '_' +'Tanh')(decoder)
        decoder_list.append(decoder)
        unet_generator = Model(input = [unet_input_layer], output = [decoder], name = 'UNet_Generator')


        self.logger_model.info('UNet generator summary:')
        unet_generator.summary(print_fn=self.logger_model.info)
        plot_model(unet_generator, to_file='UNet_generator.png')

        return unet_generator



    def conv_block(self,layer_input, filters, encoder_index, filter_name, is_BatchNorm = True, strides = (2,2), name = 'Encoder'):
        '''
        Each conv_block contains Conv2D(filters)-BatchNorm-LeakyReLU(0.2)
        '''
        x = Conv2D(filters, (4,4), strides = strides,
                    name = name +'_' + str(encoder_index) + '_' + filter_name,
                    padding = "same", data_format = self._data_format)(layer_input)
        if is_BatchNorm:
            if self._data_format == "channels_first":
                # if _data_format == "channels_first", set axis = 1
                x = BatchNormalization(axis = 1, name = name +'_' +str(encoder_index) + '_' +'BatchNorm')(x)
            else:
                x = BatchNormalization(axis = -1, name = name +'_' + str(encoder_index) + '_' +'BatchNorm')(x)
        x = LeakyReLU(0.2, name = name +'_' + str(encoder_index) + '_' +'LeakyReLU')(x)
        return x



    def up_conv_block(self,layer_input, layer_connected, filters, decoder_index, filter_name, is_BatchNorm = True, dropout = False, concatenate = True, name = 'Decoder'):
        '''
        Each upconv_block contains Conv2D(filters)-UpSample2D-BatchNorm-(Dropout(0.5))-ReLU
        '''
        x = Conv2D(filters,(4,4), name = name + '_' + str(decoder_index) + '_' + filter_name,
                    padding = "same", data_format = self._data_format)(layer_input)
        x = UpSampling2D(size=(2,2), name = name +'_' + str(decoder_index) + '_' +'UpSampling2D')(x)
        if is_BatchNorm:
            if self._data_format == "channels_first":
                # if _data_format == "channels_first", set axis = 1
                x = BatchNormalization(axis = 1, name = name +'_' + str(decoder_index) + '_' +'BatchNorm')(x)
            else:
                x = BatchNormalization(axis = -1, name = name +'_' + str(decoder_index) + '_' +'BatchNorm')(x)
        if dropout:
            x = Dropout(0.5, seed = 1, name = name +'_' + str(decoder_index) + '_' +'Dropout')(x)
        if concatenate:
            if self._data_format == "channels_first":
                x = Concatenate(axis = 1, name = name +'_' + str(decoder_index) + '_' +'Concatenate')([x, layer_connected])
            else:
                x = Concatenate(axis = -1, name = name +'_' + str(decoder_index) + '_' +'Concatenate')([x, layer_connected])
        x = Activation("relu", name = 'Decoder_' + str(decoder_index) + '_' +'ReLU')(x)
        return x


    def deconv_block(self,layer_input, layer_connected, filters, decoder_index, filter_name, is_BatchNorm = True, dropout = False, concatenate = True, name = 'Decoder'):
        '''
        Each deconv_block contains Conv2DTranspose(filters)-BatchNorm-(Dropout(0.5))-ReLU
        NOTE: Deconv2D is called Conv2DTranspose in latest keras
        '''
        x = Conv2DTranspose(filters,(4,4), strides = (2,2), padding = 'same',
                            name = name +'_' + str(decoder_index) + '_' + filter_name,
                            data_format = self._data_format)(layer_input)

        if is_BatchNorm:
            if self._data_format == "channels_first":
                # if data_format == "channels_first", set axis = 1
                x = BatchNormalization(axis = 1, name = name +'_' + str(decoder_index) + '_' +'BatchNorm')(x)
            else:
                x = BatchNormalization(axis = -1, name = name + '_' + str(decoder_index) + '_' +'BatchNorm')(x)
        if dropout:
            x = Dropout(0.5, seed = 1, name = name +'_' + str(decoder_index) + '_' +'Dropout')(x)
        if concatenate:
            if self._data_format == "channels_first":
                x = Concatenate(axis = 1, name = name +'_' + str(decoder_index) + '_' +'Concatenate')([x, layer_connected])
            else:
                x = Concatenate(axis = -1, name = name +'_' + str(decoder_index) + '_' +'Concatenate')([x, layer_connected])
        x = Activation("relu", name = name +'_' + str(decoder_index) + '_' +'ReLU')(x)
        return x




    def build_discriminator(self):
        '''
        Creator PatchGAN discriminator only penalizing
        structure at the scale of patches. The discriminator
        classifies each NxN patch in the image is real or fake.
        The discriminator convolutationally run across the image,
        averaging all the responses to provide the ultimate output
        of D

        Implement for 70x70 discriminator
        Architecture: C64-C128-C256-C512
                      Flatten()-Dense("sigmoid")

        CHANGE: have overlapping for the patches to mitigate the patch boundary effect for generated image

        :param patch_dim: ()
        :param num_patch: ()
        '''


        '''
        def minb_disc(x):
            diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
            abs_diffs = K.sum(K.abs(diffs), 2)
            x = K.sum(K.exp(-abs_diffs), 2)

            return x

        def lambda_output(input_shape):
            return input_shape[:2]
        '''

        # ------------------------------------------------------------
        # Build discriminator C64-C128-C256-C512-Flatten()-Dense("sigmoid")
        # ------------------------------------------------------------

        # Define filters for different patch size,
        # Exception: for pixelGAN, Conv2D layer filter kernel (1,1)
        switcher = {1: ["C64","C128"],
            16: ["C64","C128"],
            32: ["C64","C128","C256"],
            64: ["C64","C128","C256","C512"],
            128: ["C64","C128","C256","C512","C512"],
            256: ["C64","C128","C256","C512","C512","C512"]}
        if self._data_format == "channels_first":
            patch_width = self.patch_dim[1]
        else:
            patch_width = self.patch_dim[0]
        discriminator_filters = switcher.get(patch_width, "Invalid patch size")
        text = 'Patch size: ({0},{1}) \n Discriminator structure: '.format(self.patch_dim[0],self.patch_dim[1])
        text = text + '_'.join(discriminator_filters)
        print(text)

        # Construct discriminator using conv_block
        discriminator_input_layer =  Input(shape=self.patch_dim, name="Discriminator_input")
        discriminator = discriminator_input_layer
        discriminator_list = [discriminator]

        for discriminator_i, discriminator_filter in enumerate(discriminator_filters):
            print('layer {} filter {}'.format(discriminator_i, discriminator_filter))
            filters = int(discriminator_filter[1:])
            if discriminator_i == 0:
                # First layer without BatchNorm
                discriminator = self.conv_block(discriminator, filters, encoder_index = discriminator_i, filter_name = discriminator_filter, is_BatchNorm = False, name = 'Discriminator')
            else:
                discriminator = self.conv_block(discriminator, filters, encoder_index = discriminator_i, filter_name = discriminator_filter, name = 'Discriminator')
            discriminator_list.append(discriminator)

        # Last 2 layers: Flatten() + Dense("sigmoid")
        discriminator_flatten = Flatten(name = 'Discriminator_' + str(discriminator_i+1) + '_' +'Flatten')(discriminator)
        discriminator_list.append(discriminator_flatten)
        discriminator_sigmoid = Dense(2, activation = "softmax", name = 'Discriminator_' + str(discriminator_i+2) + '_' +'Sigmoid')(discriminator_flatten)

        PatchGAN = Model(input = [discriminator_input_layer], output = [discriminator_sigmoid, discriminator_flatten], name = 'PatchGAN')
        print('discriminator backbone summary:')
        print(PatchGAN.summary())

        # ---------------
        # Build PatchGAN
        # ---------------

        # Iterate over all patch to generate input list (all the same as the discriminator_input layer)
        patch_input_list = [Input(shape=self.patch_dim, name="Discriminator_patch_"+str(i_patch)+'_input') for i_patch in range(self.num_patch)]
        # Iterate over all patch input list to get tensor after discriminator
        patch_discriminator_output_list = [PatchGAN(patch_input)[0] for patch_input in patch_input_list]

        print('patch_discriminator_output_list: {}'.format(patch_discriminator_output_list))
        # Average patch_discriminator_output_list
        if len(patch_discriminator_output_list) > 1:
            average_out = average(patch_discriminator_output_list)
        else:
            average_out = patch_discriminator_output_list[0]


        discriminator = Model(input=patch_input_list, output=[average_out], name='discriminator_PatchGAN')
        print('PatchGAN discriminator summary:')
        print(discriminator.summary())

        return discriminator
