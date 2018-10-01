# test using cpu only
cpu = False # set to True if no cpu present
cuda="0"    # which gpu to use if any (for multi-gpu setups)

# type of network to be trained,
# can be bnn: binary with float activation,
#        full-bnn: binary with binary activation,
#        qbnn: binary with quantized activation (abits is used here)
# similar for tnn (ternary weights -1/0/+1) (tnn, full-tnn, qtnn)
# similar for qnn (weights quantized according to wbits):
#        qnn: float activations, full-qnn: activations according to abits
network_type = 'full-qnn'

# bits can be None, 2, 4, 8 , whatever
bits=None
wbits = 4 # how many bits for the weights
abits = 4 # how many bits for the activation
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
data_augmentation=False

#regularization
kernel_regularizer=1e-4
kernel_initializer='he_normal'
activity_regularizer=0.

# width and depth (ONLY for VGG)
nla=1
nfa=64
nlb=1
nfb=128
nlc=1
nfc=256

# ONLY for resnet
nres=3      # size of resnet = 6*nres+2

#learning rate decay, factor => LR *= factor (IGNORED DURING RESNETS!!!)
decay_at_epoch = [0, 8, 12 ]
factor_at_epoch = [1, .1, .1]
kernel_lr_multiplier = 10

# debug and logging
progress_logging = 2 # can be 0 = no std logging, 1 = progress bar logging, 2 = one log line per epoch
epochs = 200
batch_size = 128
lr = 1e-3         #LR during 80 first epochs of resnet
decay = 0.000025

date="00/00/0000"

# important paths
out_wght_path = './weights/{}_{}_{}b_{}b_{}_{}_{}_{}_{}_{}.hdf5'.format(dataset,network_type,abits,wbits,nla,nfa,nlb,nfb,nlc,nfc)
tensorboard_name = '{}_{}_{}b_{}b_{}_{}_{}_{}_{}_{}.hdf5'.format(dataset,network_type,abits,wbits,nla,nfa,nlb,nfb,nlc,nfc)
