import sys
import itertools
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Activation, BatchNormalization
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy, mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.activations import softmax

def compile(model, loss):
    model.compile(loss=loss,
        optimizer=Adam(),
        metrics=[categorical_accuracy, binary_accuracy])

def double_models(classes, len_byte_vector):
    name = 'C64_16_2pr_C32_4_2pr_C64_32_2pr_F_D' + str(classes)
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(64, (16,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(64, (32,), strides=2, padding='same', activation='relu')(last)
    last = Flatten()(last)
    last = Dense(classes)(last)

    lastcat = Activation('softmax')(last)
    modelcat = Model(l0, lastcat, name=name + '_cat')
    compile(modelcat, 'categorical_crossentropy')

    lastmse = Activation('sigmoid')(last)
    modelmse = Model(l0, lastmse, name=name + '_mse')
    compile(modelmse, 'mse')
    return modelcat, modelmse


def C64_16_2pr_C32_4_2pr_C64_32_2pr_F_D(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name + str(classes) + '_cat'
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(64, (16,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(64, (32,), strides=2, padding='same', activation='relu')(last)
    last = Flatten()(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model(l0, last, name=myfuncname)
    compile(model, loss)
    return model

def C64_16_2pBA_C32_4_2pBA_C64_32_2pBA_F_D(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name + str(classes) + '_cat'
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(64, (16,), strides=2, padding='same')(last)
    last = BatchNormalization()(last)
    last = Activation('relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same')(last)
    last = BatchNormalization()(last)
    last = Activation('relu')(last)
    last = Conv1D(64, (32,), strides=2, padding='same')(last)
    last = BatchNormalization()(last)
    last = Activation('relu')(last)
    last = Flatten()(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model(l0, last, name=myfuncname)
    compile(model, loss)
    return model

def C256_16_16_L128_D(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name + str(classes) + '_cat'
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(256, (16,), strides=16)(last)
    last = LSTM(128)(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model(l0, last, name=myfuncname)
    compile(model, loss)
    return model

def C64_16_2pr_5C32_4_2pr_C64_32_2pr_F_D(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name + str(classes) + '_cat'
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(64, (16,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(64, (32,), strides=2, padding='same', activation='relu')(last)
    last = Flatten()(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model(l0, last, name=myfuncname)
    compile(model, loss)
    return model

# def C64_16_1pr_5C32_4_1pr_C64_32_1pr_L64_D(classes, len_byte_vector, activation, loss):
#     myfuncname = sys._getframe().f_code.co_name + str(classes) + '_cat'
#     last = l0 = Input(shape=(512,len_byte_vector))
#     last = Conv1D(64, (16,), strides=1, padding='same', activation='relu')(last)
#     last = Conv1D(32, (4,), strides=1, padding='same', activation='relu')(last)
#     last = Conv1D(32, (4,), strides=1, padding='same', activation='relu')(last)
#     last = Conv1D(32, (4,), strides=1, padding='same', activation='relu')(last)
#     last = Conv1D(32, (4,), strides=1, padding='same', activation='relu')(last)
#     last = Conv1D(32, (4,), strides=1, padding='same', activation='relu')(last)
#     last = Conv1D(64, (32,), strides=1, padding='same', activation='relu')(last)
#     last = LSTM(64)(last)
#     last = Dense(classes)(last)
#     last = Activation(activation)(last)
#     model = Model(l0, last, name=myfuncname)
#     compile(model, loss)
#     return model

def C32_4_2PR_C64_32_2PR_F_D(classes, len_byte_vector, activation, loss):
    myfuncname = sys._getframe().f_code.co_name + str(classes) + '_cat'
    last = l0 = Input(shape=(512,len_byte_vector))
    last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
    last = Conv1D(64, (32,), strides=2, padding='same', activation='relu')(last)
    last = Flatten()(last)
    last = Dense(classes)(last)
    last = Activation(activation)(last)
    model = Model(l0, last, name=myfuncname)
    compile(model, loss)
    return model

# def C64_64_8pr_C64_16_16pr_L64_D(classes, len_byte_vector, activation, loss):
#     myfuncname = sys._getframe().f_code.co_name + str(classes) + '_cat'
#     last = l0 = Input(shape=(512,len_byte_vector))
#     last = Conv1D(64, (16,), strides=2, padding='same', activation='relu')(last)
#     last = Conv1D(32, (4,), strides=2, padding='same', activation='relu')(last)
#     last = LSTM(64)(last)
#     last = Dense(classes)(last)
#     last = Activation(activation)(last)
#     model = Model(l0, last, name=myfuncname)
#     compile(model, loss)
#     return model
