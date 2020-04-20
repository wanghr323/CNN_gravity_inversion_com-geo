

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from numpy import shape
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import matplotlib as m
import DataReader as dr
import inv_model as inv_model
import inv_plot as inv_p
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import multi_gpu_model

#the following paths should be changed to your own path!!!!!!!
input_model = '/home/yc/test_data/models_dataset.txt'
input_anomal = '/home/yc/test_data/anomal_dataset.txt'
input_dxdydz = '/home/yc/test_data/dxdy_dataset.txt'
input_model_test = '/home/yc/test_data/models_dataset_t.txt'
input_anomal_test = '/home/yc/test_data/anomal_dataset_t.txt'
input_dxdydz_test = '/home/yc/test_data/dxdy_dataset_t.txt'
model_weight_save_path = '/home/yc/test_data/3Dmodel_weight.h5'
model_plot_path = '/home/yc/test_data/modelplot.png'

Nx = 20
Ny = 20
Nz = 10

d_model, d_anomal = dr.load_3Ddata(input_model, input_anomal, Nx, Ny, Nz)
dxdy, dz = dr.load_3Ddata_dxdydz(input_dxdydz)


d_model_t, d_anomal_t = dr.load_3Ddata(input_model_test, input_anomal_test, Nx, Ny, Nz)
dxdy_t, dz_t = dr.load_3Ddata_dxdydz(input_dxdydz_test)


model = inv_model.inv_model2()
print(model.summary())
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: %d' % strategy.num_replicas_in_sync)  
with strategy.scope():
    model.compile(optimizer='adam',
                  loss='mean_squared_logarithmic_error',
                  metrics=['mape'])
    history = model.fit({'anomal_input': d_anomal, 'dxdy_input':dxdy},
                            d_model, epochs=1000, batch_size=32)
    test_loss = model.evaluate({'anomal_input': d_anomal_t, 'dxdy_input':dxdy_t},
                           d_model_t, verbose=2)
model.save_weights(model_weight_save_path)
# model.load_weights(model_save_path)
print(history.history)
d_model_p = model.predict({'anomal_input': d_anomal_t, 'dxdy_input':dxdy_t})
# # draw
box_num1 = 80
#
inv_p.plot_model(d_model_t[box_num1, :, :, :])
inv_p.plot_model(d_model_p[box_num1, :, :, :])
inv_p.plot_model(d_model_p[box_num1+1, :, :, :])
inv_p.plot_model(d_model_p[box_num1+2, :, :, :])
inv_p.plot_model(d_model_p[box_num1+3, :, :, :])

inv_p.history_plot(history)
plot_model(model, to_file=model_plot_path)




