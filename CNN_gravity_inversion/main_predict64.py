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
import Datatransfer as ds
import os
#choose the gpu number
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.MirroredStrategy()
print('Number of device :%d ' % strategy.num_replicas_in_sync)



#the following paths should be changed to your own path!!!!!!!
input_pre_anomal = '/home/yc/test_data/anomal_one.txt'
model_weight_save_path = '/home/yc/test_data/3Dmodel_weight3by3.h5'
model_save_path = '/home/yc/test_data/3Dpredict_model_one.txt'

Nx_ori = 64
Ny_ori = 64
Nz_ori = 24
begin_x = 0
end_x = 630
begin_y = 0
end_y = 630
end_z = 23

Nx = 64
Ny = 64
Nz = 24

anomal = dr.load_pre_anomal(input_pre_anomal, Nx_ori, Ny_ori, Nx, Ny)
[dx, dy, dz] = [(end_x - begin_x)/(Nx-1), (end_y - begin_y)/(Ny-1), end_z/(Nz-1)]
dxdy = np.array([dx/dz, dy/dz])
dxdy = np.resize(dxdy, (1, 2))
anomal = ds.anomal_tran_deltz(anomal, dz)

with strategy.scope():
    model2 = inv_model.inv_model7_3by3()
    print(model2.summary())
    model2.load_weights(model_weight_save_path)
    print('starting predict\n')
    d_model_p = model2.predict({'anomal_input': anomal, 'dxdy_input': dxdy})
    print('loading pre_model\n')
    print('plot result\n')

inv_p.plot_anomal(anomal[0, 0, :, :])
inv_p.plot_model_64(d_model_p[0, :, :, :])

d_model_p = np.resize(d_model_p, (64*64*24, 1))

np.savetxt(model_save_path, d_model_p, fmt='%.07f', newline='\n')
