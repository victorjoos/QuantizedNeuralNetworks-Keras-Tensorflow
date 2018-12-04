# test using cpu only
cpu = False

# type of network to be trained, can be bnn, full-bnn, qnn, full-qnn, tnn, full-tnn
network_type = 'full-qnn'
# bits can be None, 2, 4, 8 , whatever
bits=None
wbits = 4
abits = 4
# finetune an be false or true
finetune = False

architecture = 'RESNET'
# architecture = 'VGG'
dataset='CIFAR-10'
# dataset='MNIST'

if dataset == 'CIFAR-10':
    dim=32
    channels=3
else:
    dim=28
    channels=1
classes=10
data_augmentation=True

#regularization
kernel_regularizer=0.
kernel_initializer='glorot_uniform'
activity_regularizer=0.

# width and depth
nla=1
nfa=64
nlb=1
nfb=128
nlc=1
nfc=256

nres=3
pfilt=1
cuda="0"

#learning rate decay, factor => LR *= factor
decay_at_epoch = [0, 8, 12 ]
factor_at_epoch = [1, .1, .1]
kernel_lr_multiplier = 10

# debug and logging
progress_logging = 2 # can be 0 = no std logging, 1 = progress bar logging, 2 = one log line per epoch
epochs = 500
batch_size = 128
lr = 0.1
decay = 0.000025

date="00/00/0000"

# important paths
out_wght_path = './weights/{}_{}_{}b_{}b_{}_{}_{}_{}_{}_{}.hdf5'.format(dataset,network_type,abits,wbits,nla,nfa,nlb,nfb,nlc,nfc)
tensorboard_name = '{}_{}_{}b_{}b_{}_{}_{}_{}_{}_{}.hdf5'.format(dataset,network_type,abits,wbits,nla,nfa,nlb,nfb,nlc,nfc)
