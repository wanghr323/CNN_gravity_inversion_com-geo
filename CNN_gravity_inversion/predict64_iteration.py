from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from numpy import shape
import numpy as np
import DataReader as dr
import inv_model as inv_model
import inv_plot as inv_p
import Datatransfer as ds
import cppforward as gforward
def mean_relative_error(y_true, y_pred):
    relative_error = np.average(np.abs(y_true - y_pred) / y_true)
    return relative_error


def tran_fuction(model,tran_min,tran_max):
    model[model < tran_min] = tran_min
    model[model > tran_max] = tran_max
    return model

input_pre_anomal = '/home/yc/test_data/anomal_one.txt'
input_pre_model = '/home/yc/test_data/model_one.txt'
model_weight_save_path = '/home/yc/test_data/3Dmodel_weight3by3.h5'
model_save_path = '/home/yc/test_data/3Dpredict_model.txt'




#basic parameter
times = 10
Nx_ori = 64
Ny_ori = 64
begin_x = 0
end_x = 630
begin_y = 0
end_y = 630
end_z = 23
Nx = 64
Ny = 64
Nz = 24
tran_min = 0
tran_max = 1
[dx, dy, dz] = [(end_x - begin_x)/(Nx-1), (end_y - begin_y)/(Ny-1), end_z/(Nz-1)]
dxdy = np.array([dx/dz, dy/dz])
dxdy = np.resize(dxdy, (1, 2))

#initial the forward matrix Va
Va = gforward.gravity_forward_Va(Nx,Ny,Nz,dx,dy,dz)

#load the network model
inv_model = inv_model.inv_model7_3by3()
print(inv_model.summary())
inv_model.load_weights(model_weight_save_path)

#some initial matrix:vobs, model, anomal_predict
vobs = dr.load_pre_anomal(input_pre_anomal, Nx_ori, Ny_ori, Nx, Ny)
vobs = ds.anomal_tran_deltz(vobs, dz)
vobs = np.reshape(vobs, (Nx*Ny))
model = np.zeros(Nx*Ny*Nz)
anomal_predict = np.zeros(Nx*Ny)

#show the real model
print('loading pre_model\n')
d_model_real = dr.load_pre_model(input_pre_model, 64, 64, 24)
print('plot model\n')
inv_p.plot_model_64(d_model_real[0, :, :, :])

#show the vobs anomal
vobs_draw = dr.load_pre_anomal(input_pre_anomal, Nx_ori, Ny_ori, Nx, Ny)
inv_p.plot_anomal(vobs_draw[0, 0, :, :])

error = 1000
#starting the iteration
for i in range (1, times):
    flag = 0
    print('the ', i, ' th times iteration ')
    pos_anomal = (vobs-anomal_predict)
    pos_anomal[pos_anomal < 0] = 0
    pos_model_delt = inv_model.predict({'anomal_input': np.reshape(pos_anomal,(1,1,Nx,Ny)),'dxdy_input': dxdy})
    pos_model_delt = np.reshape(pos_model_delt, Nz*Nx*Ny)
    model_p = model+pos_model_delt
    model_p = tran_fuction(model_p, tran_min, tran_max)
    anomal_predict_p = gforward.gravity_forward(model_p, Va, Nx, Ny, Nz)
    error_p = mean_relative_error(vobs, anomal_predict_p)
    while flag < 5 and error_p > error:
        # if error_p < error:
        flag = flag+1
        model_p = model+pos_model_delt/(2**flag)
        model_p = tran_fuction(model_p, tran_min, tran_max)
        anomal_predict_p = gforward.gravity_forward(model_p, Va, Nx, Ny, Nz)
        error_p = mean_relative_error(vobs, anomal_predict_p)
        print('flag = ', flag, 'pos shrink once')
    flag = 0
    if error_p < error:
        model = model_p
        anomal_predict = anomal_predict_p
        error = error_p
    else:
        print('did not go')

    #calculate the misfit(anomal,anomal_predict)
    print('after pos predict:  error:   ', error)
    # draw the anomal_predict #draw the model_predict
    title = 'the '+str(i)+' pos result'
    inv_p.plot_anomal(np.reshape(anomal_predict, (1, 1, Nx, Ny))[0, 0, :, :], title)
    inv_p.plot_model_64(np.reshape(model, (1, Nz, Nx, Ny))[0, :, :, :], title)

    neg_anomal = -(vobs-anomal_predict)
    neg_anomal[neg_anomal < 0] = 0
    neg_model_delt = inv_model.predict({'anomal_input': np.reshape(neg_anomal,(1,1,Nx,Ny)),'dxdy_input': dxdy})
    neg_model_delt = np.reshape(neg_model_delt, (Nz*Nx*Ny))
    model_p = model - neg_model_delt
    model_p = tran_fuction(model_p, tran_min, tran_max)
    anomal_predict_p = gforward.gravity_forward(model_p, Va, Nx, Ny, Nz)
    error_p = mean_relative_error(vobs, anomal_predict_p)

    while flag < 5 and error_p > error:
        # if error_p < error:
        flag = flag+1
        model_p = model-neg_model_delt/(2**flag)
        model_p = tran_fuction(model_p, tran_min, tran_max)
        anomal_predict_p = gforward.gravity_forward(model_p, Va, Nx, Ny, Nz)
        error_p = mean_relative_error(vobs, anomal_predict_p)
        print('flag = ', flag, 'neg shrink once')
    flag = 0
    if error_p < error:
        model = model_p
        anomal_predict = anomal_predict_p
        error = error_p
    else:
        print('did not go')

    # calculate the misfit(anomal,anomal_predict)
    print('after neg predict:  error:   ', error)
    title = 'the ' + str(i) + ' neg result'
    # draw the anomal_predict  # draw the model_predict
    inv_p.plot_anomal(np.reshape(anomal_predict, (1, 1, Nx, Ny))[0, 0, :, :], title)
    inv_p.plot_model_64(np.reshape(model, (1, Nz, Nx, Ny))[0, :, :, :], title)




np.savetxt(model_save_path, model, fmt='%.07f', newline='\n')
