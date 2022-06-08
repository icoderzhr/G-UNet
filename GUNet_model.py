import math
from keras import Input
from keras.layers import Conv2D, Dropout, Concatenate, DepthwiseConv2D, MaxPooling2D, UpSampling2D, concatenate, \
    Activation, BatchNormalization, Lambda, add, SeparableConv2D, SeparableConv1D, multiply, AveragePooling2D
from keras.models import Model
from keras.utils import plot_model


def slices(x,channel):
    y = x[:,:,:,:channel]
    return y

def multiply(x,excitation):
    scale = x * excitation
    return scale

def GhostModule(x,outchannels,ratio,convkernel,dwkernel,padding='same',strides=1,data_format='channels_last',
                use_bias=False,activation=None):
    conv_out_channel = math.ceil(outchannels*1.0/ratio)
    x = Conv2D(int(conv_out_channel),(convkernel,convkernel),strides=(strides,strides),padding=padding,data_format=data_format,
               activation=activation,use_bias=use_bias, kernel_initializer='random_uniform')(x)
    if(ratio==1):
        return x

    dw = DepthwiseConv2D(dwkernel,strides,padding=padding,depth_multiplier=ratio-1,data_format=data_format,
                         activation=activation,use_bias=use_bias,kernel_initializer='random_uniform')(x)
    #dw = dw[:,:,:,:int(outchannels-conv_out_channel)]
    #dw = Lambda(slices,arguments={'channel':int(outchannels-conv_out_channel)})(dw)
    x = Concatenate(axis=-1)([x,dw])
    return x


def GhostBottleneck(x,dwkernel,strides,exp,out,ratio):
    x1 = DepthwiseConv2D(dwkernel,strides,padding='same',depth_multiplier=ratio-1,data_format='channels_last',
                         activation=None,use_bias=False,kernel_initializer='random_uniform')(x)
    x1 = BatchNormalization(axis=-1)(x1)
    x1 = Conv2D(out,(1,1),strides=(1,1),padding='same',data_format='channels_last',
               activation=None,use_bias=False,kernel_initializer='random_uniform')(x1)
    x1 = BatchNormalization(axis=-1)(x1)
    y1 = GhostModule(x,exp,ratio,1,3)
    y1 = BatchNormalization(axis=-1)(y1)
    y1 = Activation('relu')(y1)
    #weight_1 = Lambda(lambda x: x * 0.1)
    #y1 = weight_1(y1)
    #y1 = add([x1,y1])
    if(strides>1):
        y = DepthwiseConv2D(dwkernel,strides,padding='same',depth_multiplier=ratio-1,data_format='channels_last',
                         activation=None,use_bias=False)(y1)
        y = BatchNormalization(axis=-1)(y)
        y = Activation('relu')(y)
    y2 = GhostModule(y1,out,ratio,1,3)
    y2 = BatchNormalization(axis=-1)(y2)
    #weight_2 = Lambda(lambda x: x * 0.1)
    #y2 = weight_2(y2)
    #2 = add([y1, y2])
    y = add([x1,y2])
    return y

def Ghost_UNet (input_size=(128, 128, 1)):
    inputs = Input(input_size)
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation=None,
               use_bias=False)(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x1 = GhostBottleneck(x, 3, 1, 16, 16, 2)
    pool1 = AveragePooling2D(pool_size=(2, 2))(x1)
    x2 = GhostBottleneck(pool1, 3, 1, 32, 32, 2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(x2)
    x3 = GhostBottleneck(pool2, 3, 1, 64, 64, 2)
    pool3 = AveragePooling2D(pool_size=(2, 2))(x3)
    x4 = GhostBottleneck(pool3, 3, 1, 128, 128, 2)
    #drop4 = Dropout(0.5)(x4)
    pool4 = AveragePooling2D(pool_size=(2, 2))(x4)
    x5 = GhostBottleneck(pool4, 3, 1, 256, 256, 2)
    #drop5 = Dropout(0.5)(x5)

    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='random_uniform')(
        UpSampling2D(size=(2, 2))(x5))
    merge6 = concatenate([x4, up6], axis=3)
    x6 = GhostBottleneck(merge6, 3, 1, 128, 128, 2)

    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='random_uniform')(
        UpSampling2D(size=(2, 2))(x6))
    merge7 = concatenate([x3, up7], axis=3)
    x7 = GhostBottleneck(merge7, 3, 1, 64, 64, 2)

    up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='random_uniform')(
        UpSampling2D(size=(2, 2))(x7))
    merge8 = concatenate([x2, up8], axis=3)
    x8 = GhostBottleneck(merge8, 3, 1, 32, 32, 2)

    up9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='random_uniform')(
        UpSampling2D(size=(2, 2))(x8))
    merge9 = concatenate([x1, up9], axis=3)
    x9 = GhostBottleneck(merge9, 3, 1, 16, 16, 2)

    x9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='random_uniform')(x9)
    outputs = Conv2D(1, 1,activation= 'relu', kernel_initializer='random_uniform')(x9)

    model = Model(inputs=inputs, outputs=outputs)
    # plot_model(model,to_file=os.path.join('weight', "GhostNet_model.png"), show_shapes=True)
    # plot_model(model,'GhostNet_model.png', show_shapes=True)
    #model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=3e-4), metrics=['accuracy'])
    return model

#浮点数计算
import keras.backend as K
import tensorflow as tf
def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


def main():
    model = Ghost_UNet()
    print(get_flops(model))
    model.summary()
    plot_model(model, to_file='./GU_model.png', show_shapes=True)



if __name__ == '__main__':
    main()
