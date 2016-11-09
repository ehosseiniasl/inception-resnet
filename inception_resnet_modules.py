'''
keras implemention for Inception-Resnet-V4 Network
Paper: [https://arxiv.org/pdf/1602.07261v2.pdf]

By: Ehsan Hosseini-Asl
Date: November 2016

'''
from keras.layers import Input, Convolution2D, Dense, Flatten, LSTM, MaxPooling2D, Dropout, Merge, AveragePooling2D, Activation

def stem_v4(x):
    '''
    Inception-v4 stem module (no valid convolution)
    '''
    h1 = Convolution2D(32, 3, 3, border='same', strides=(2, 2))(x) # x shape = (1, h, w)
    h2 = Convolution2D(64, 3, 3)(h1)
    h3_max_pool = MaxPooling2D(pool_shape=(3, 3), strides=(2, 2))(h2)
    h3_conv = Convolution2D(96, 3, 3, strides=(2, 2))
    h4 = Merge([h3_max_pool, h3_conv], mode='concat', concat_axis=1) # h4 shape = (160, h/4, w/4)
    h5_1 = Convolution2D(64, 1, 1, border_mode='same')(h4)
    h6_1 = Convolution2D(96, 3, 3, border_mode='same')(h5_1)
    h5_2 = Convolution2D(64, 1, 1, border_mode='same')(h4)
    h6_2 = Convolution2D(64, 7, 1, border_mode='same')(h5_2)
    h7_2 = Convolution2D(64, 1, 7, border_mode='same')(h6_2)
    h8_2 = Convolution2D(96, 3, 3, border_mode='same')(h7_2)
    h9 = Merge([h6_1, h8_2], mode='concat', concat_axis=1) # h9 shape = (h/4, w/4, 192)
    h10_1 = Convolution2D(192, 3, 3, border_mode='same', strides=(2, 2))(h9)
    h10_2 = MaxPooling2D(pool_shape=(3, 3), strides=(2, 2))(h9)
    output = Merge([h10_1, h10_2], mode='concat', concat_axis=1) # h11 shape = (384, h/8, w/8)
    return output


def stem_resnet(x):
    '''
    Inception-resnet stem module (no valid convolution)
    '''
    h1 = Convolution2D(32, 3, 3, border='same', strides=(2, 2))(x) # x shape = (1, h, w)
    h2 = Convolution2D(32, 3, 3, border_mode='same')(h1)
    h3 = Convolution2D(64, 3, 3, border_mode='same')(h2)
    h4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h3)
    h5 = Convolution2D(80, 1, 1, border_mode='same')(h4)
    h6 = Convolution2D(192, 3, 3, border_mode='same')(h5)
    output = Convolution2D(256, 3, 3, border_mode='same', strides=(2, 2))(h6)
    return output

def inception_A(x):
    h1_1 = Convolution2D(64, 1, 1, border_mode='same')(x)
    h2_1 = Convolution2D(96, 3, 3, border_mode='same')(h1_1)
    h3_1 = Convolution2D(96, 3, 3, border_mode='same')(h2_1)
    h1_2 = Convolution2D(64, 1, 1, border_mode='same')(x)
    h2_2 = Convolution2D(96, 3, 3, border_mode='same')(h1_2)
    h1_3 = Convolution2D(96, 1, 1, border_mode='same')(x)
    h1_4 = AveragePooling2D(pool_shape=(3, 3))(x)
    h2_4 = Convolution2D(96, 1, 1, border_mode='same')(h1_4)
    output = Merge([h3_1, h2_2, h1_3, h2_4], mode='concat', concat_axis=1) # output shape = (384, h, w)
    return output


def inception_resnet_A(x):
    h1 = Activation(activation='relu')(x)
    h2_1 = Convolution2D(32, 1, 1, border_mode='same')(h1)
    h3_1 = Convolution2D(32, 3, 3, border_mode='same')(h2_1)
    h4_1 = Convolution2D(32, 3, 3, border_mode='same')(h3_1)
    h2_2 = Convolution2D(32, 1, 1, border_mode='same')(h1)
    h3_2 = Convolution2D(32, 3, 3, border_mode='same')(h2_2)
    h2_3 = Convolution2D(96, 1, 1, border_mode='same')(h1)
    residual = Merge([h4_1, h3_2, h2_3], mode='concat', concat_axis=1) # output shape = (384, h, w)
    output = Merge([h1, residual], mode='sum')
    return output

def reduction_A(x, k, l, m, n):
    h1_1 = Convolution2D(k, 1, 1, border_mode='same')(x)
    h2_1 = Convolution2D(l, 3, 3, border_mode='same')(h1_1)
    h3_1 = Convolution2D(m, 3, 3, strides=(2, 2), border_mode='same')(h2_1)
    h1_2 = Convolution2D(n, 3, 3, strides=(2, 2), border_mode='same')(x)
    h1_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    output = Merge([h3_1, h1_2, h1_3], mode='concat', concat_axis=1)
    return output

def inception_B(x):
    h1_1 = Convolution2D(192, 1, 1, border_mode='same')(x)
    h2_1 = Convolution2D(192, 1, 7, border_mode='same')(h1_1)
    h3_1 = Convolution2D(224, 7, 1, border_mode='same')(h2_1)
    h4_1 = Convolution2D(224, 1, 7, border_mode='same')(h3_1)
    h5_1 = Convolution2D(256, 7, 1, border_mode='same')(h4_1)
    h1_2 = Convolution2D(192, 1, 1, border_mode='same')(x)
    h2_2 = Convolution2D(224, 1, 7, border_mode='same')(h1_2)
    h3_2 = Convolution2D(256, 1, 7, border_mode='same')(h2_2)
    h1_3 = Convolution2D(384, 1, 1, border_mode='same')(x)
    h1_4 = AveragePooling2D(pool_shape=(3, 3))(x)
    h2_4 = Convolution2D(128, 1, 1, border_mode='same')(h1_4)
    output = Merge([h5_1, h3_2, h1_3, h2_4], mode='concat', concat_axis=1) # output shape = (384, h, w)
    return output

def reduction_B(x):
    h1_1 = Convolution2D(256, 1, 1, border_mode='same')(x)
    h2_1 = Convolution2D(256, 1, 7, border_mode='same')(h1_1)
    h3_1 = Convolution2D(320, 7, 1, border_mode='same')(h2_1)
    h4_1 = Convolution2D(320, 3, 3, strides=(2, 2), border_mode='same')(h3_1)
    h1_2 = Convolution2D(192, 1, 1, border_mode='same')(x)
    h2_2 = Convolution2D(192, 3, 3, strides=(2, 2), border_mode='same')(h1_2)
    h1_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    output = Merge([h4_1, h2_2, h1_3], mode='concat', concat_axis=1)
    return output


def reduction_resnet_B(x):
    h1_1 = Convolution2D(256, 1, 1, border_mode='same')(x)
    h2_1 = Convolution2D(256, 3, 3, border_mode='same')(h1_1)
    h3_1 = Convolution2D(256, 3, 3, strides=(2, 2), border_mode='same')(h2_1)
    h1_2 = Convolution2D(256, 1, 1, border_mode='same')(x)
    h2_2 = Convolution2D(256, 3, 3, strides=(2, 2), border_mode='same')(h1_2)
    h1_3 = Convolution2D(256, 1, 1, border_mode='same')(x)
    h2_3 = Convolution2D(384, 3, 3, strides=(2, 2), border_mode='same')(h1_3)
    h1_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    output = Merge([h3_1, h2_2, h2_3, h1_4], mode='concat', concat_axis=1)
    return output

def inception_resnet_B(x):
    '''
    input depth should be equal to output depth
    '''
    h1 = Activation(activation='relu')(x)
    h2_1 = Convolution2D(128, 1, 1, border_mode='same')(h1)
    h3_1 = Convolution2D(128, 1, 7, border_mode='same')(h2_1)
    h4_1 = Convolution2D(128, 7, 1, border_mode='same')(h3_1)
    h2_2 = Convolution2D(128, 1, 1, border_mode='same')(h1)
    h3 = Merge([h4_1, h2_2], mode='concat', concat_axis=1)
    residual = Convolution2D(896, 1, 1, morder_mode='same')(h3)
    sum = Merge([h1, residual], mode='sum')
    output = Activation(activation='relu')(sum)
    return output


def inception_C(x):
    h1_1 = Convolution2D(384, 1, 1, border_mode='same')(x)
    h2_1 = Convolution2D(448, 1, 3, border_mode='same')(h1_1)
    h3_1 = Convolution2D(512, 3, 1, border_mode='same')(h2_1)
    h4_1_1 = Convolution2D(256, 1, 3, border_mode='same')(h3_1)
    h4_2_1 = Convolution2D(256, 3, 1, border_mode='same')(h3_1)
    h1_2 = Convolution2D(384, 1, 1, border_mode='same')(x)
    h2_1_2 = Convolution2D(256, 3, 1, border_mode='same')(h1_2)
    h2_2_2 = Convolution2D(256, 1, 3, border_mode='same')(h1_2)
    h1_3 = Convolution2D(256, 1, 1, border_mode='same')(x)
    h1_4 = AveragePooling2D(pool_size=(3, 3))(x)
    h2_4 = Convolution2D(256, 1, 1, border_mode='same')(h1_4)
    output = Merge([h4_1_1, h4_2_1, h2_1_2, h2_2_2, h1_3, h2_4], mode='concat', concat_azis=1)
    return output


def inception_resnet_C(x):
    '''
    input depth should be equal to output depth
    '''
    h1 = Activation(activation='relu')(x)
    h2_1 = Convolution2D(192, 1, 1, border_mode='same')(h1)
    h3_1 = Convolution2D(192, 1, 3, border_mode='same')(h2_1)
    h4_1 = Convolution2D(192, 3, 1, border_mode='same')(h3_1)
    h2_2 = Convolution2D(192, 1, 1, border_mode='same')(h1)
    h3 = Merge([h4_1, h2_2], mode='concat', concat_axis=1)
    residual = Convolution2D(1792, 1, 1, morder_mode='same')(h3)
    sum = Merge([h1, residual], mode='sum')
    output = Activation(activation='relu')(sum)
    return output


