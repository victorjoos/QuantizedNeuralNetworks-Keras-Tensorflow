from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.losses import squared_hinge
import os
import argparse
import keras.backend as K
import numpy as np
from models.model_factory import build_model
from utils.config_utils import Config
from utils.load_data import load_dataset
import os

# parse arguments
parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('-c', '--config_path', type=str,
                default=None, help='Configuration file')
parser.add_argument('-o' ,'--override',action='store',nargs='*',default=[])

arguments = parser.parse_args()
override_dir = {}

for s in arguments.override:
    s_s = s.split("=")
    k = s_s[0].strip()
    v = "=".join(s_s[1:]).strip()
    override_dir[k]=v
arguments.override = override_dir


cfg = arguments.config_path
cf = Config(cfg, cmd_args = arguments.override)


# if necessary, only use the CPU for debugging
if cf.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=cf.cuda

# ## Construct the network
print('Construct the Network\n')
model = build_model(cf)

print('loading data\n')
train_data, val_data, test_data = load_dataset(cf.dataset, cf)

print('setting up the network and creating callbacks\n')
checkpoint = ModelCheckpoint(cf.out_wght_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
tensorboard = TensorBoard(log_dir='./logs/' + str(cf.tensorboard_name), histogram_freq=0, write_graph=True, write_images=False)
callbacks = [checkpoint, tensorboard]
if cf.architecture == "VGG":
    early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, mode='min', verbose=1)

    def scheduler(epoch):
        if epoch in cf.decay_at_epoch:
            index = cf.decay_at_epoch.index(epoch)
            factor = cf.factor_at_epoch[index]
            lr = K.get_value(model.optimizer.lr)
            IT = train_data.X.shape[0]/cf.batch_size
            current_lr = lr * (1./(1.+cf.decay*epoch*IT))
            K.set_value(model.optimizer.lr,current_lr*factor)
            print('\nEpoch {} updates LR: LR = LR * {} = {}\n'.format(epoch+1,factor, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)

    lr_decay = LearningRateScheduler(scheduler)
    adam = Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=cf.decay)
    callbacks += [lr_decay]
    loss = squared_hinge

elif cf.architecture=="RESNET":
    def lr_schedule(epoch):
        lr = cf.lr
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks += [lr_scheduler, lr_reducer]
    adam = Adam(lr=lr_schedule(0))
    loss = 'categorical_crossentropy'


print('compiling the network\n')
model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])

if cf.finetune:
    print('Load previous weights\n')
    model.load_weights(cf.out_wght_path)
else:
    print('No weights preloaded, training from scratch\n')

print('(re)training the network\n')
if not cf.data_augmentation:
    model.fit(train_data.X,train_data.y,
                batch_size=cf.batch_size,
                epochs=cf.epochs,
                verbose=cf.progress_logging,
                shuffle=True,
                callbacks=callbacks,
                validation_data=(val_data.X,val_data.y))
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=False, # divide inputs by std of dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        zca_epsilon=1e-06, # epsilon for ZCA whitening
        rotation_range=0, # randomly rotate images in the range (deg 0 to 180)
        width_shift_range=0.1, # randomly shift images horizontally
        height_shift_range=0.1, # randomly shift images vertically
        shear_range=0., # set range for random shear
        zoom_range=0., # set range for random zoom
        channel_shift_range=0., # set range for random channel shifts
        fill_mode='nearest', # set mode for filling points outside the input boundaries
        cval=0., # value used for fill_mode = "constant"
        horizontal_flip=True, # randomly flip images
        vertical_flip=False, # randomly flip images
        rescale=None, # set rescaling factor (applied before any other transformation)
        preprocessing_function=None, # set function that will be applied on each input
        data_format=None, # image data format, either "channels_first" or "channels_last"
        validation_split=0.0) # fraction of images reserved for validation (strictly between 0 and 1)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(train_data.X)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(train_data.X,train_data.y, batch_size=cf.batch_size),
                        epochs=cf.epochs,
                        verbose=cf.progress_logging,
                        callbacks=callbacks,
                        validation_data=(val_data.X,val_data.y))


print('Done\n')
