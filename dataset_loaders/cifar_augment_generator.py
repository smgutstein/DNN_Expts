from keras.preprocessing.image import ImageDataGenerator

def get_generator(self):
    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False)  # randomly flip images

    datagen.fit(self.X_train)
    datagen = datagen.flow(self.X_train, self.Y_train, batch_size=self.batch_size)
    datagen.numImages = self.Y_train.shape[0]
    datagen.batches_per_epoch = datagen.numImages //self.batch_size

    # create generator for test data - like generator for train data, only without image augmentation
    dummy_datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=False,  # randomly flip images
                    vertical_flip=False)  # randomly flip images
    info = "Data generator to augment data"

    dummy_datagen.fit(self.X_test)
    dummy_datagen = dummy_datagen.flow(self.X_test, self.Y_test, batch_size=self.batch_size)
    dummy_datagen.numImages = self.Y_test.shape[0]
    dummy_datagen.batches_per_epoch = dummy_datagen.numImages // self.batch_size

    return (datagen, dummy_datagen, info)

