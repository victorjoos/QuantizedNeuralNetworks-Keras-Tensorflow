from keras.models import Sequential, Model
from keras import regularizers
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import numpy as np

from layers.quantized_layers import QuantizedConv2D,QuantizedDense
from layers.quantized_ops import quantized_relu as quantize_op
from layers.binary_layers import BinaryConv2D, BinaryDense
from layers.binary_ops import binary_tanh
from layers.ternary_layers import TernaryConv2D, TernaryDense
from layers.ternary_ops import ternary_tanh

def build_model(cf):
    def quantized_relu(x):
        return quantize_op(x,nb=cf.abits)


    H = 1.
    if cf.network_type =='float':
        Conv = lambda s, f, i=None, c=None: Conv2D(
            kernel_size=(s, s), filters=f, strides=(1, 1), padding='same',
            activation='linear',
            kernel_regularizer=l2(cf.kernel_regularizer),
            **({'input_shape': (i,i,c)} if i!=None else {})
        )
        Fc = lambda o: Dense(o, use_bias=False)
        Act = lambda: LeakyReLU()

    elif cf.network_type in ['qnn', 'full-qnn']:
        Conv = lambda s, f,i=None,c=None: QuantizedConv2D(
            kernel_size=(s, s), H=1, nb=cf.wbits, filters=f, strides=(1, 1),
            padding='same', activation='linear',
            kernel_regularizer=l2(cf.kernel_regularizer),
            kernel_lr_multiplier=cf.kernel_lr_multiplier,
            **({'input_shape': (i,i,c)} if i!=None else {})
        )
        Fc = lambda o: QuantizedDense(o, use_bias=False)

        if cf.network_type=='qnn':
            Act = lambda: LeakyReLU()
        else: # full-qnn
            Act = lambda: Activation(quantized_relu)

    elif cf.network_type in ['bnn', 'qbnn', 'full-bnn']:
        Conv = lambda s, f, i=None, c=None: BinaryConv2D(
            kernel_size=(s, s), H=1, filters=f, strides=(1, 1), padding='same',
             activation='linear', kernel_regularizer=l2(cf.kernel_regularizer),
             kernel_lr_multiplier=cf.kernel_lr_multiplier,
             **({'input_shape': (i,i,c)} if i!=None else {})
         )
        Fc = lambda o: BinaryDense(o, use_bias=False)

        if cf.network_type=='bnn':
            Act = lambda: LeakyReLU()
        elif cf.network_type=='qbnn':
            Act = lambda: Activation(quantized_relu)
        else: #full-bnn
            Act = lambda: Activation(binary_tanh)

    elif cf.network_type in ['tnn', 'qtnn', 'full-tnn']:
        Conv = lambda s, f, i=None, c=None: TernaryConv2D(
            kernel_size=(s, s), H=1, filters=f, strides=(1, 1), padding='same',
            activation='linear', kernel_regularizer=l2(cf.kernel_regularizer),
            kernel_lr_multiplier=cf.kernel_lr_multiplier,
            **({'input_shape': (i,i,c)} if i!=None else {})
        )
        Fc = lambda o: TernaryDense(o, use_bias=False)

        if cf.network_type=='tnn':
            Act = lambda: LeakyReLU()
        elif cf.network_type=='qtnn':
            Act = lambda: Activation(quantized_relu)
        else: #full-tnn
            Act = lambda: Activation(ternary_tanh)

    else:
        print('wrong network type, the supported network types in this repo are float, qnn, full-qnn, bnn and full-bnn')


    model = Sequential()
    model.add(Conv(3, cf.nfa,cf.dim,cf.channels))
    model.add(BatchNormalization(momentum=0.1,epsilon=0.0001))
    model.add(Act())
    # block A
    for i in range(0,cf.nla-1):
        model.add(Conv(3, cf.nfa))
        model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
        model.add(Act())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # block B
    for i in range(0,cf.nlb):
        model.add(Conv(3, cf.nfb))
        model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
        model.add(Act())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # block C
    for i in range(0,cf.nlc):
        model.add(Conv(3, cf.nfc))
        model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
        model.add(Act())
    model.add(MaxPooling2D(pool_size=(2, 2)))


    # Dense Layer
    model.add(Flatten())
    model.add(Fc(cf.classes))
    model.add(BatchNormalization(momentum=0.1,epsilon=0.0001))

    # In[5]:
    model.summary()

    return model




def load_weights(model, weight_reader):
    weight_reader.reset()

    for i in range(len(model.layers)):
        if 'conv' in model.layers[i].name:
            if 'batch' in model.layers[i + 1].name:
                norm_layer = model.layers[i + 1]
                size = np.prod(norm_layer.get_weights()[0].shape)

                beta = weight_reader.read_bytes(size)
                gamma = weight_reader.read_bytes(size)
                mean = weight_reader.read_bytes(size)
                var = weight_reader.read_bytes(size)

                weights = norm_layer.set_weights([gamma, beta, mean, var])

            conv_layer = model.layers[i]
            if len(conv_layer.get_weights()) > 1:
                bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel, bias])
            else:
                kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel])
    return model
