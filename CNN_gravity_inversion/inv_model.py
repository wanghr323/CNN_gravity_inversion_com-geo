from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

###hte following models are the models I have tried,but i didn't delete it, but we finally choose the inv_model7_3by3()
def inv_model():
    model = models.Sequential()
    model.add(layers.Conv2D(4, (2, 2), input_shape=(1, 20, 20), strides=(2, 2), activation='relu',
                            data_format="channels_first"))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first'))
    model.add(layers.Conv2D(10, (2, 2), activation='relu', data_format="channels_first"))
    model.add(layers.Conv2D(10, (2, 2), activation='relu', padding='same', data_format="channels_first"))
    model.add(layers.Flatten())
    model.add(layers.Dense(4000, activation='sigmoid'))
    # model.add(layers.Flatten(input_shape=(1, 20, 20)))
    # model.add(layers.Dense(4000, activation='sigmoid'))
    model.add(layers.Reshape((10, 20, 20)))
    return model


def inv_model2():
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(4, (2, 2), activation='relu', strides=(2, 2), data_format="channels_first")(anomal_input)
    x = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)
    x = layers.Conv2D(10, (2, 2), activation='relu', data_format="channels_first")(x)
    x = layers.Conv2D(10, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Flatten()(x)

    y = layers.concatenate([dxdy_input, x])
    y = layers.Dense(160, activation='relu')(y)
    y = layers.Dense(160, activation='relu')(y)

    y = layers.Dense(4000, activation='sigmoid')(y)
    model_output = layers.Reshape((10, 20, 20), name='model_output')(y)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model3():
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(8, (2, 2), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)
    x = layers.Conv2D(16, (2, 2), activation='relu',padding='same', data_format="channels_first")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)
    x = layers.Conv2D(32, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)
    x = layers.Conv2D(64, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)
    x = layers.Flatten()(x)

    y = layers.concatenate([dxdy_input, x])
    y = layers.Dense(400, activation='relu')(y)
    y = layers.Dense(400, activation='relu')(y)

    y = layers.Dense(98304, activation='sigmoid')(y)
    model_output = layers.Reshape((24, 64, 64), name='model_output')(y)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model4():
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(4, (2, 2), activation='relu', strides=(2, 2), data_format="channels_first")(anomal_input)
    x = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)
    x = layers.Conv2D(10, (2, 2), activation='relu', data_format="channels_first")(x)
    x = layers.Conv2D(10, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Flatten()(x)

    y = layers.concatenate([dxdy_input, x])
    y1 = layers.Dense(40, activation='relu')(y)
    y2 = layers.Dense(40, activation='relu')(y)
    y3 = layers.Dense(40, activation='relu')(y)
    y4 = layers.Dense(40, activation='relu')(y)
    y5 = layers.Dense(40, activation='relu')(y)
    y6 = layers.Dense(40, activation='relu')(y)
    y7 = layers.Dense(40, activation='relu')(y)
    y8 = layers.Dense(40, activation='relu')(y)
    y9 = layers.Dense(40, activation='relu')(y)
    y10 = layers.Dense(40, activation='relu')(y)




    # y1 = layers.Dense(200, activation='relu')(y1)
    # y2 = layers.Dense(200, activation='relu')(y2)
    # y3 = layers.Dense(200, activation='relu')(y3)
    # y4 = layers.Dense(200, activation='relu')(y4)
    # y5 = layers.Dense(200, activation='relu')(y5)
    # y6 = layers.Dense(200, activation='relu')(y6)
    # y7 = layers.Dense(200, activation='relu')(y7)
    # y8 = layers.Dense(200, activation='relu')(y8)
    # y9 = layers.Dense(200, activation='relu')(y9)
    # y10 = layers.Dense(200, activation='relu')(y10)

    y1 = layers.Dense(400, activation='sigmoid')(y1)
    y2 = layers.Dense(400, activation='sigmoid')(y2)
    y3 = layers.Dense(400, activation='sigmoid')(y3)
    y4 = layers.Dense(400, activation='sigmoid')(y4)
    y5 = layers.Dense(400, activation='sigmoid')(y5)
    y6 = layers.Dense(400, activation='sigmoid')(y6)
    y7 = layers.Dense(400, activation='sigmoid')(y7)
    y8 = layers.Dense(400, activation='sigmoid')(y8)
    y9 = layers.Dense(400, activation='sigmoid')(y9)
    y10 = layers.Dense(400, activation='sigmoid')(y10)

    y = layers.concatenate([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10])

    # y = layers.Dense(4000, activation='sigmoid')(y)
    model_output = layers.Reshape((10, 20, 20), name='model_output')(y)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model5():
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(2, (2, 2), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(6, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(10, (2, 2), activation='sigmoid', padding='same', data_format="channels_first")(x)

    y = layers.Dense(10, activation='relu')(dxdy_input)
    y = layers.Dense(25, activation='relu')(y)
    y = layers.Dense(50, activation='relu')(y)
    y = layers.Dense(4000, activation='sigmoid')(y)
    y = layers.Reshape((10, 20, 20))(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((10, 20, 20), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model5_train_model():
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(2, (2, 2), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(6, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(10, (2, 2), activation='sigmoid', padding='same', data_format="channels_first")(x)

    y = layers.Dense(10, activation='relu', trainable=False)(dxdy_input)
    y = layers.Dense(25, activation='relu', trainable=False)(y)
    y = layers.Dense(50, activation='relu', trainable=False)(y)
    y = layers.Dense(4000, activation='sigmoid', trainable=False)(y)
    y = layers.Reshape((10, 20, 20))(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((10, 20, 20), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model5_train_dxdy():
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(2, (2, 2), activation='relu', padding='same', trainable=False, data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (2, 2), activation='relu', padding='same', trainable=False, data_format="channels_first")(x)
    x = layers.Conv2D(6, (2, 2), activation='relu', padding='same', trainable=False, data_format="channels_first")(x)
    x = layers.Conv2D(8, (2, 2), activation='relu', padding='same', trainable=False, data_format="channels_first")(x)
    x = layers.Conv2D(10, (2, 2), activation='sigmoid', padding='same', trainable=False, data_format="channels_first")(x)

    y = layers.Dense(10, activation='relu')(dxdy_input)
    y = layers.Dense(25, activation='relu')(y)
    y = layers.Dense(50, activation='relu')(y)
    y = layers.Dense(4000, activation='sigmoid')(y)
    y = layers.Reshape((10, 20, 20))(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((10, 20, 20), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model6():
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(2, (2, 2), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(6, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(10, (2, 2), activation='sigmoid', padding='same', data_format="channels_first")(x)

    y = layers.Dense(10, activation='relu')(dxdy_input)
    y = layers.Dense(25, activation='relu')(y)
    y = layers.Dense(50, activation='relu')(y)
    y = layers.Dense(400, activation='relu')(y)

    y = layers.Reshape((1, 20, 20))(y)
    y = layers.Conv2D(5, (2, 2), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(10, (2, 2), activation='relu', padding='same', data_format="channels_first")(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((10, 20, 20), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model6_train_model():
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(2, (3, 3), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(6, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(10, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    y = layers.Dense(10, trainable=False, activation='relu')(dxdy_input)
    y = layers.Dense(25, trainable=False, activation='relu')(y)
    y = layers.Dense(50, trainable=False, activation='relu')(y)
    y = layers.Dense(400, trainable=False, activation='relu')(y)

    y = layers.Reshape((1, 20, 20))(y)
    y = layers.Conv2D(5, (3, 3), activation='relu', trainable=False, padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(10, (3, 3), activation='relu', trainable=False, padding='same', data_format="channels_first")(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((10, 20, 20), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model6_train_dxdy():
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(2, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(6, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(10, (3, 3), trainable=False, activation='sigmoid', padding='same', data_format="channels_first")(x)

    y = layers.Dense(10, activation='relu')(dxdy_input)
    y = layers.Dense(25, activation='relu')(y)
    y = layers.Dense(50, activation='relu')(y)
    y = layers.Dense(400, activation='relu')(y)

    y = layers.Reshape((1, 20, 20))(y)
    y = layers.Conv2D(5, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(10, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((10, 20, 20), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model



def inv_model7_relu():
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(2, (3, 3), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    y = layers.Dense(4, activation='relu')(dxdy_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model7_3by3():
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(2, (3, 3), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    y = layers.Dense(4, activation='relu')(dxdy_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model

def inv_model7_5by5():
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(2, (5, 5), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (5, 5), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (5, 5), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (5, 5), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (5, 5), activation='sigmoid', padding='same', data_format="channels_first")(x)

    y = layers.Dense(4, activation='relu')(dxdy_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (5, 5), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (5, 5), activation='sigmoid', padding='same', data_format="channels_first")(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model

def inv_model7_7by7():
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(2, (7, 7), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (7, 7), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (7, 7), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (7, 7), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (7, 7), activation='sigmoid', padding='same', data_format="channels_first")(x)

    y = layers.Dense(4, activation='relu')(dxdy_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (7, 7), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (7, 7), activation='sigmoid', padding='same', data_format="channels_first")(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model7_9by9():
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(2, (9, 9), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (9, 9), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (9, 9), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (9, 9), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (9, 9), activation='sigmoid', padding='same', data_format="channels_first")(x)

    y = layers.Dense(4, activation='relu')(dxdy_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (9, 9), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (9, 9), activation='sigmoid', padding='same', data_format="channels_first")(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model7_dz():
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    dxdydz_input = Input(shape=(3), name='dxdydz_input')

    x = layers.Conv2D(2, (3, 3), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    y = layers.Dense(12, activation='relu')(dxdydz_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdydz_input], outputs=model_output)
    return model


def inv_model7_train_model():
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(2, (3, 3), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    y = layers.Dense(4, trainable=False, activation='relu')(dxdy_input)
    y = layers.Dense(64, trainable=False, activation='relu')(y)
    y = layers.Dense(128, trainable=False, activation='relu')(y)
    y = layers.Dense(4096, trainable=False, activation='relu')(y)

    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model7_train_dxdy():
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    dxdy_input = Input(shape=(2), name='dxdy_input')

    x = layers.Conv2D(2, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (3, 3), trainable=False, activation='sigmoid', padding='same', data_format="channels_first")(x)

    y = layers.Dense(4, activation='relu')(dxdy_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)

    z = layers.Multiply()([x, y])
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model