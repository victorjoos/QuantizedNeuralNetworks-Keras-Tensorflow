from keras.models import Sequential
from keras.layers import MaxPooling2D, BatchNormalization, Flatten
from keras.regularizers import l2

def Vgg(Conv, Act, Fc, cf):
    input_shape = (cf.dim,cf.dim,cf.channels)
    def make_conv(ksize, filters, **kwargs):
        return Conv(
            kernel_size=(ksize, ksize), filters=filters, strides=(1,1), padding='same',
            kernel_initializer=cf.kernel_initializer,
            kernel_regularizer=l2(cf.kernel_regularizer),
            **kwargs
        )
    model = Sequential()
    model.add(make_conv(3, cf.nfa, input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.1,epsilon=0.0001))
    model.add(Act())
    # block A
    for i in range(0,cf.nla-1):
        model.add(make_conv(3, cf.nfa))
        model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
        model.add(Act())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # block B
    for i in range(0,cf.nlb):
        model.add(make_conv(3, cf.nfb))
        model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
        model.add(Act())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # block C
    for i in range(0,cf.nlc):
        model.add(make_conv(3, cf.nfc))
        model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
        model.add(Act())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dense Layer
    model.add(Flatten())
    model.add(Fc(cf.classes))
    model.add(BatchNormalization(momentum=0.1,epsilon=0.0001))

    return model
