from keras.preprocessing.image import ImageDataGenerator

def get_generator(x_train, y_train, batch_size):
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
    info = "Data generator to augment data"

    datagen.fit(x_train)
    datagen = datagen.flow(x_train, y_train, batch_size=batch_size)

    return (datagen, info)
