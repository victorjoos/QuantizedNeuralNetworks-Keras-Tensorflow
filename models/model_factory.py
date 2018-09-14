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

from models.resnet import ResNet18
from models.vgg import Vgg

def build_model(cf):
    def quantized_relu(x):
        return quantize_op(x,nb=cf.abits)


    H = 1.
    if cf.network_type =='float':
        Conv = Conv2D
        Fc = Dense
        Act = lambda: LeakyReLU()

    elif cf.network_type in ['qnn', 'full-qnn']:
        Conv = lambda **kwargs: QuantizedConv2D(H=1, nb=cf.wbits, **kwargs)
        Fc = QuantizedDense

        if cf.network_type=='qnn':
            Act = lambda: LeakyReLU()
        else: # full-qnn
            Act = lambda: Activation(quantized_relu)

    elif cf.network_type in ['bnn', 'qbnn', 'full-bnn']:
        Conv = lambda **kwargs: BinaryConv2D(H=1, **kwargs)
        Fc = BinaryDense

        if cf.network_type=='bnn':
            Act = lambda: LeakyReLU()
        elif cf.network_type=='qbnn':
            Act = lambda: Activation(quantized_relu)
        else: #full-bnn
            Act = lambda: Activation(binary_tanh)

    elif cf.network_type in ['tnn', 'qtnn', 'full-tnn']:
        Conv = lambda **kwargs: TernaryConv2D(H=1, **kwargs)
        Fc = TernaryDense

        if cf.network_type=='tnn':
            Act = lambda: LeakyReLU()
        elif cf.network_type=='qtnn':
            Act = lambda: Activation(quantized_relu)
        else: #full-tnn
            Act = lambda: Activation(ternary_tanh)

    else:
        raise ValueError('wrong network type, the supported network types in this repo are float, qnn, full-qnn, bnn and full-bnn')

    if cf.architecture=="VGG":
        model = Vgg(Conv, Act, Fc, cf)
    elif cf.architecture=="RESNET":
        model = ResNet18(Conv, Act, Fc, cf)
    else:
        raise ValueError("Error: type " + str(cf.architecture) + " is not supported")

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
