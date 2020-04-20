from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import tensorflow as tf
from numpy import shape
from PIL import Image
import matplotlib.pyplot as plt


def load_data(f_model, f_anomal, Nx, Ny):
    model = np.loadtxt(f_model, delimiter=" ")
    model = np.resize(model, (model.shape[0], Nx, Ny, 1))
    anomal = np.loadtxt(f_anomal, delimiter=" ")
    # anomal = np.resize(anomal, (model.shape[0], Nx, 1, 1))
    print("model\n", model)
    print("anomal\n", anomal)
    return model, anomal


def load_3Ddata(f_model, f_anomal, Nx, Ny, Nz):
    model = np.loadtxt(f_model, delimiter=" ")

    print('shape of f_model:\n', model.shape[0], model.shape[1])
    model = np.resize(model, (model.shape[0], Nz, Nx, Ny))
    anomal = np.loadtxt(f_anomal, delimiter=" ")
    anomal = np.resize(anomal, (anomal.shape[0], 1, Nx, Ny))
    return model, anomal


def load_3Ddata_dxdydz(f_dxdydz):
    dxdydz = np.loadtxt(f_dxdydz, delimiter=" ")
    print("chushi:\n", dxdydz)
    dxdydz = np.resize(dxdydz, (dxdydz.shape[0], 3))
    dz = dxdydz[:, 2:3]
    dxdy = dxdydz[:, 0:2]
    dxdy = np.true_divide(dxdy, dz)
    print("dxdy: ", dxdy)

    return dxdy, dz

def load_3Ddata_dxdydz_dz(f_dxdydz):
    dxdydz = np.loadtxt(f_dxdydz, delimiter=" ")
    print("chushi:\n", dxdydz)
    dxdydz = np.resize(dxdydz, (dxdydz.shape[0], 3))

    return dxdydz


def load_pre_anomal(f_anomal, Nx, Ny, final_Nx, final_Ny):
    anomal = np.loadtxt(f_anomal)
    anomal = np.resize(anomal, (Nx, Ny))
    im = Image.fromarray(anomal)
    im_res = im.resize((final_Nx, final_Ny), Image.ANTIALIAS)
    anomal = np.array(im_res)
    anomal = np.resize(anomal, (1, 1, final_Nx, final_Ny))
    return anomal


def load_pre_model(f_model, Nx, Ny, Nz):
    print("loading...pre_model")
    model = np.loadtxt(f_model, delimiter=" ")
    print("resizing...pre_model")
    model = np.resize(model, (1, Nz, Nx, Ny))

    return model